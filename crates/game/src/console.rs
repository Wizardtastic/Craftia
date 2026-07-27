//! Developer console with cursor editing, history, tab-completion, and
//! scrollback output. Separate from the chat system — different visual
//! style, different keybind, richer input model (multi-line, cursor
//! movement).

use std::collections::{HashMap, VecDeque};

/// Maximum number of lines the console scrollback can hold.
const MAX_SCROLLBACK: usize = 512;

/// Maximum number of history entries.
const MAX_HISTORY: usize = 100;

// -----------------------------------------------------------------
// ConsoleLine — single-line editing buffer with cursor
// -----------------------------------------------------------------

/// A single editing buffer with a movable cursor. Supports insert,
/// delete (backspace and forward-delete), and cursor movement.
pub struct ConsoleLine {
    chars: Vec<char>,
    /// Cursor position: index into `chars` (0 = before first char,
    /// `chars.len()` = after last char).
    cursor: usize,
}

impl ConsoleLine {
    pub fn new() -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
        }
    }

    pub fn insert_char(&mut self, ch: char) {
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
    }

    /// Backspace: delete the character before the cursor.
    pub fn delete_before_cursor(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    /// Forward-delete: delete the character at the cursor.
    pub fn delete_at_cursor(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    pub fn cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn cursor_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
    }

    pub fn cursor_home(&mut self) {
        self.cursor = 0;
    }

    pub fn cursor_end(&mut self) {
        self.cursor = self.chars.len();
    }

    /// Return the full text of this line.
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    /// Cursor position (char index).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Replace the contents of this line and move cursor to end.
    pub fn set_text(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
    }
}

impl Default for ConsoleLine {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------
// CompletionNode — trie for tab-completion
// -----------------------------------------------------------------

/// A trie node for prefix-based tab completion.
struct CompletionNode {
    children: HashMap<char, CompletionNode>,
    /// Set if this node is the end of a registered command.
    is_end: bool,
    /// Full command string at this endpoint.
    full_command: Option<String>,
}

impl CompletionNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
            full_command: None,
        }
    }

    /// Insert a command string into the trie.
    pub fn insert(&mut self, command: &str) {
        let mut node = self;
        for ch in command.chars() {
            node = node.children.entry(ch).or_insert_with(CompletionNode::new);
        }
        node.is_end = true;
        node.full_command = Some(command.to_string());
    }

    /// Returns all completions for the given prefix, plus the longest
    /// common prefix of all matches.
    pub fn complete(&self, prefix: &str) -> CompletionResult {
        let mut node = self;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(child) => node = child,
                None => {
                    return CompletionResult {
                        matches: Vec::new(),
                        common_prefix: prefix.to_string(),
                    };
                }
            }
        }

        let mut matches = Vec::new();
        Self::collect_matches(node, &mut matches);

        let common_prefix = if matches.is_empty() {
            prefix.to_string()
        } else {
            Self::longest_common_prefix(&matches, prefix.len())
        };

        CompletionResult {
            matches,
            common_prefix,
        }
    }

    fn collect_matches(node: &CompletionNode, out: &mut Vec<String>) {
        if let Some(ref cmd) = node.full_command {
            out.push(cmd.clone());
        }
        for child in node.children.values() {
            Self::collect_matches(child, out);
        }
    }

    fn longest_common_prefix(matches: &[String], min_len: usize) -> String {
        if matches.is_empty() {
            return String::new();
        }
        let first = &matches[0];
        let mut end = first.len();
        for m in &matches[1..] {
            end = end.min(
                first
                    .chars()
                    .zip(m.chars())
                    .take_while(|(a, b)| a == b)
                    .count(),
            );
        }
        end = end.max(min_len);
        first.chars().take(end).collect()
    }
}

pub struct CompletionResult {
    pub matches: Vec<String>,
    pub common_prefix: String,
}

// -----------------------------------------------------------------
// DeveloperConsole
// -----------------------------------------------------------------

/// The developer console. Maintains its own input line(s), scrollback
/// output, command history, and tab-completion trie. The engine owns
/// this struct and drives it from key events; the console itself
/// never touches the ECS or the world.
pub struct DeveloperConsole {
    pub open: bool,
    /// Current input line.
    line: ConsoleLine,
    /// Scrollback: older output at the front.
    scrollback: VecDeque<String>,
    /// Scroll offset (0 = newest, >0 = scrolled up).
    scroll_offset: usize,
    /// Command history (for up/down arrows).
    history: VecDeque<String>,
    history_index: Option<usize>,
    /// Tab-completion trie of all known commands.
    completion_trie: CompletionNode,
    /// Blink state for cursor rendering (toggled each ~500ms by the
    /// engine via `tick_cursor`).
    cursor_visible: bool,
    /// Accumulated time since last cursor blink toggle.
    cursor_blink_timer: f64,
}

impl DeveloperConsole {
    pub fn new() -> Self {
        Self {
            open: false,
            line: ConsoleLine::new(),
            scrollback: VecDeque::with_capacity(MAX_SCROLLBACK),
            scroll_offset: 0,
            history: VecDeque::with_capacity(MAX_HISTORY),
            history_index: None,
            completion_trie: CompletionNode::new(),
            cursor_visible: true,
            cursor_blink_timer: 0.0,
        }
    }

    /// Open the console and set up a fresh input line.
    pub fn open(&mut self) {
        self.open = true;
        self.line.clear();
        self.history_index = None;
        self.scroll_offset = 0;
        self.cursor_visible = true;
        self.cursor_blink_timer = 0.0;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.line.clear();
        self.history_index = None;
        self.scroll_offset = 0;
    }

    /// Submit the current input for execution. Returns the raw text
    /// (the engine dispatches it). The input is recorded in history.
    pub fn submit(&mut self) -> String {
        let text = self.line.text();
        self.line.clear();
        self.history_index = None;
        self.scroll_offset = 0;

        if text.is_empty() {
            return String::new();
        }

        // Record in history (no duplicates of the last entry).
        if self.history.front().map(|s| s.as_str()) != Some(&text) {
            self.history.push_front(text.clone());
            while self.history.len() > MAX_HISTORY {
                self.history.pop_back();
            }
        }

        text
    }

    /// Add a line to the scrollback output.
    pub fn println(&mut self, msg: String) {
        self.scrollback.push_back(msg);
        while self.scrollback.len() > MAX_SCROLLBACK {
            self.scrollback.pop_front();
        }
        // Auto-scroll to bottom on new output.
        self.scroll_offset = 0;
    }

    // -- Input methods --

    pub fn insert_char(&mut self, ch: char) {
        if !ch.is_control() {
            self.line.insert_char(ch);
            self.reset_cursor_blink();
        }
    }

    pub fn backspace(&mut self) {
        self.line.delete_before_cursor();
        self.reset_cursor_blink();
    }

    pub fn delete(&mut self) {
        self.line.delete_at_cursor();
        self.reset_cursor_blink();
    }

    pub fn cursor_left(&mut self) {
        self.line.cursor_left();
        self.reset_cursor_blink();
    }

    pub fn cursor_right(&mut self) {
        self.line.cursor_right();
        self.reset_cursor_blink();
    }

    /// History: go back (older).
    pub fn cursor_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_index {
            None => 0,
            Some(i) if i + 1 < self.history.len() => i + 1,
            _ => return,
        };
        self.history_index = Some(next);
        self.line.set_text(&self.history[next]);
    }

    /// History: go forward (newer).
    pub fn cursor_down(&mut self) {
        match self.history_index {
            None => {}
            Some(0) => {
                self.history_index = None;
                self.line.clear();
            }
            Some(i) => {
                let next = i - 1;
                self.history_index = Some(next);
                self.line.set_text(&self.history[next]);
            }
        }
    }

    pub fn cursor_home(&mut self) {
        self.line.cursor_home();
        self.reset_cursor_blink();
    }

    pub fn cursor_end(&mut self) {
        self.line.cursor_end();
        self.reset_cursor_blink();
    }

    /// Tab-complete the current prefix. Inserts the common prefix
    /// and prints matches to scrollback if ambiguous.
    pub fn tab_complete(&mut self) {
        let text = self.line.text();
        let trimmed = text.trim_start();
        if trimmed.is_empty() {
            return;
        }
        let result = self.completion_trie.complete(trimmed);
        if result.matches.is_empty() {
            return;
        }
        if result.matches.len() == 1 {
            self.line.set_text(&format!("{} ", result.matches[0]));
        } else if result.common_prefix.len() > trimmed.len() {
            self.line.set_text(&result.common_prefix);
        } else {
            self.println(format!("Commands: {}", result.matches.join(", ")));
        }
        self.reset_cursor_blink();
    }

    /// Register commands for tab-completion. Call once at startup with
    /// all known `/` commands.
    pub fn register_commands(&mut self, commands: &[&str]) {
        for cmd in commands {
            self.completion_trie.insert(cmd);
        }
    }

    // -- Rendering helpers --

    /// Current input line text.
    pub fn current_line_text(&self) -> String {
        self.line.text()
    }

    /// Cursor position within the current line (char index).
    pub fn cursor_pos(&self) -> usize {
        self.line.cursor()
    }

    /// Whether the cursor blink is in the visible phase.
    pub fn cursor_visible(&self) -> bool {
        self.cursor_visible
    }

    /// Advance the cursor blink timer. Call once per frame with the
    /// frame delta in seconds.
    pub fn tick_cursor(&mut self, dt: f64) {
        self.cursor_blink_timer += dt;
        if self.cursor_blink_timer >= 0.5 {
            self.cursor_blink_timer -= 0.5;
            self.cursor_visible = !self.cursor_visible;
        }
    }

    fn reset_cursor_blink(&mut self) {
        self.cursor_visible = true;
        self.cursor_blink_timer = 0.0;
    }

    /// Return up to `max` scrollback lines, newest first, accounting
    /// for scroll offset.
    pub fn visible_lines(&self, max: usize) -> Vec<&str> {
        let total = self.scrollback.len();
        if total == 0 {
            return Vec::new();
        }
        let end = total.saturating_sub(self.scroll_offset);
        let start = end.saturating_sub(max);
        self.scrollback.range(start..end).map(|s| s.as_str()).collect()
    }

    /// Scroll up by `n` lines.
    pub fn scroll_up(&mut self, n: usize) {
        let max_offset = self.scrollback.len().saturating_sub(1);
        self.scroll_offset = (self.scroll_offset + n).min(max_offset);
    }

    /// Scroll down by `n` lines.
    pub fn scroll_down(&mut self, n: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(n);
    }
}

impl Default for DeveloperConsole {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_line_insert_and_text() {
        let mut line = ConsoleLine::new();
        line.insert_char('h');
        line.insert_char('i');
        assert_eq!(line.text(), "hi");
        assert_eq!(line.cursor(), 2);
    }

    #[test]
    fn console_line_backspace() {
        let mut line = ConsoleLine::new();
        line.insert_char('a');
        line.insert_char('b');
        line.insert_char('c');
        line.delete_before_cursor();
        assert_eq!(line.text(), "ab");
        assert_eq!(line.cursor(), 2);
    }

    #[test]
    fn console_line_delete_at_cursor() {
        let mut line = ConsoleLine::new();
        line.insert_char('a');
        line.insert_char('b');
        line.insert_char('c');
        line.cursor_left();
        line.cursor_left();
        line.delete_at_cursor();
        assert_eq!(line.text(), "ac");
        assert_eq!(line.cursor(), 1);
    }

    #[test]
    fn console_line_cursor_movement() {
        let mut line = ConsoleLine::new();
        line.insert_char('a');
        line.insert_char('b');
        line.insert_char('c');
        line.cursor_home();
        assert_eq!(line.cursor(), 0);
        line.cursor_end();
        assert_eq!(line.cursor(), 3);
        line.cursor_left();
        line.cursor_left();
        assert_eq!(line.cursor(), 1);
        line.cursor_right();
        assert_eq!(line.cursor(), 2);
    }

    #[test]
    fn console_line_insert_at_middle() {
        let mut line = ConsoleLine::new();
        line.insert_char('a');
        line.insert_char('c');
        line.cursor_left();
        line.insert_char('b');
        assert_eq!(line.text(), "abc");
        assert_eq!(line.cursor(), 2);
    }

    #[test]
    fn completion_trie_unique() {
        let mut trie = CompletionNode::new();
        trie.insert("/help");
        trie.insert("/tp");
        let result = trie.complete("/he");
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0], "/help");
    }

    #[test]
    fn completion_trie_common_prefix() {
        let mut trie = CompletionNode::new();
        trie.insert("/time set");
        trie.insert("/time speed");
        let result = trie.complete("/ti");
        assert_eq!(result.matches.len(), 2);
        assert!(result.common_prefix.starts_with("/time "));
    }

    #[test]
    fn completion_trie_no_match() {
        let mut trie = CompletionNode::new();
        trie.insert("/help");
        let result = trie.complete("/zzz");
        assert!(result.matches.is_empty());
    }

    #[test]
    fn console_submit_records_history() {
        let mut console = DeveloperConsole::new();
        console.open();
        console.insert_char('/');
        console.insert_char('h');
        let text = console.submit();
        assert_eq!(text, "/h");
        assert_eq!(console.history.len(), 1);
    }

    #[test]
    fn console_submit_no_duplicates() {
        let mut console = DeveloperConsole::new();
        console.open();
        console.insert_char('/');
        console.submit();
        console.open();
        console.insert_char('/');
        console.submit();
        assert_eq!(console.history.len(), 1);
    }

    #[test]
    fn console_history_navigation() {
        let mut console = DeveloperConsole::new();
        console.open();
        console.line.set_text("/first");
        console.submit();
        console.open();
        console.line.set_text("/second");
        console.submit();

        console.open();
        console.cursor_up();
        assert_eq!(console.current_line_text(), "/second");
        console.cursor_up();
        assert_eq!(console.current_line_text(), "/first");
        console.cursor_down();
        assert_eq!(console.current_line_text(), "/second");
        console.cursor_down();
        assert!(console.current_line_text().is_empty());
    }

    #[test]
    fn console_scrollback_limit() {
        let mut console = DeveloperConsole::new();
        for i in 0..600 {
            console.println(format!("line {i}"));
        }
        assert!(console.scrollback.len() <= MAX_SCROLLBACK);
    }

    #[test]
    fn console_tab_complete() {
        let mut console = DeveloperConsole::new();
        console.register_commands(&["/help", "/tp", "/time set", "/time speed"]);
        console.open();
        console.line.set_text("/he");
        console.tab_complete();
        assert_eq!(console.current_line_text(), "/help ");
    }

    #[test]
    fn console_visible_lines() {
        let mut console = DeveloperConsole::new();
        console.println("a".into());
        console.println("b".into());
        console.println("c".into());
        let lines = console.visible_lines(2);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "b");
        assert_eq!(lines[1], "c");
    }
}
