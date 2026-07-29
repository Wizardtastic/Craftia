//! Undo/redo system for block edits.
//!
//! Records block changes so the player can undo (Ctrl+Z) and redo (Ctrl+Y).
//! Each action stores the list of block positions and their previous values.

use std::collections::VecDeque;

const MAX_UNDO: usize = 100;
const MAX_EDITS_PER_ACTION: usize = 65536;

/// A single block change: position and the block IDs before and after.
#[derive(Clone, Debug)]
pub struct BlockEdit {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub old_block: u16,
    pub new_block: u16,
}

/// A group of block changes that can be undone/redone atomically.
#[derive(Clone, Debug)]
pub struct EditAction {
    pub edits: Vec<BlockEdit>,
}

/// An open transaction of edits awaiting commit.
///
/// Created by [`UndoRedoState::begin_batch`]. While a batch is open, callers
/// funnel each [`BlockEdit`] through [`UndoRedoState::push_edit_batched`].
/// The batch is then either turned into a single [`EditAction`] on the undo
/// stack via [`UndoRedoState::commit_batch`], or discarded entirely via
/// [`UndoRedoState::abort_batch`].
///
/// Used by the volumetric shape rasterizer and the slash-command dispatcher
/// so that thousands of `set_block` calls collapse to a single undo step.
#[derive(Clone, Debug)]
pub struct EditBatch {
    /// Human-readable label, e.g. `"fill aabb"`. Surfaces in debugger output
    /// and the optional UI undo list.
    pub name: String,
    /// Edits accumulated so far in this batch.
    pub edits: Vec<BlockEdit>,
}

#[derive(Default)]
pub struct UndoRedoState {
    undo_stack: VecDeque<EditAction>,
    redo_stack: VecDeque<EditAction>,
    /// Currently-open batch, if any. Only one batch may be active at a time.
    active_batch: Option<EditBatch>,
}

impl UndoRedoState {
    /// Record a new action (pushes to undo stack, clears redo stack).
    pub fn push(&mut self, action: EditAction) {
        if action.edits.is_empty() {
            return;
        }
        self.push_truncated(action);
    }

    /// Internal: truncate `action.edits` to `MAX_EDITS_PER_ACTION`, push it
    /// onto the undo stack, clear the redo stack, and trim the undo stack
    /// to `MAX_UNDO`. Returns the (possibly truncated) action so that
    /// `commit_batch` can hand callers back exactly the value that landed.
    ///
    /// The single `clone()` here is bounded by `MAX_EDITS_PER_ACTION`
    /// because truncation happens *first* — callers that held a larger
    /// batch see only the truncated copy. Future changes to push semantics
    /// should land here and let both `push` and `commit_batch` share them.
    fn push_truncated(&mut self, mut action: EditAction) -> EditAction {
        if action.edits.len() > MAX_EDITS_PER_ACTION {
            action.edits.truncate(MAX_EDITS_PER_ACTION);
        }
        let returned = action.clone();
        self.undo_stack.push_front(action);
        self.redo_stack.clear();
        while self.undo_stack.len() > MAX_UNDO {
            self.undo_stack.pop_back();
        }
        returned
    }

    /// Pop the most recent undo action, if any.
    pub fn pop_undo(&mut self) -> Option<EditAction> {
        let action = self.undo_stack.pop_front()?;
        let returned = action.clone();
        self.redo_stack.push_front(action);
        Some(returned)
    }

    /// Pop the most recent redo action, if any.
    pub fn pop_redo(&mut self) -> Option<EditAction> {
        let action = self.redo_stack.pop_front()?;
        let returned = action.clone();
        self.undo_stack.push_front(action);
        Some(returned)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    // --- Batched transactions (Features 1+2+4 plumbing) -------------------

    /// Open a new edit batch with the given human-readable name. While a
    /// batch is open, [`Self::push_edit_batched`] accumulates edits into it
    /// instead of forming individual undo entries.
    ///
    /// Returns `false` (and leaves state unchanged) if a batch is already
    /// open. The boolean return makes the API ergonomic for scripting
    /// entry points where misuse should be observable rather than fatal.
    pub fn begin_batch(&mut self, name: impl Into<String>) -> bool {
        if self.active_batch.is_some() {
            return false;
        }
        self.active_batch = Some(EditBatch {
            name: name.into(),
            edits: Vec::new(),
        });
        true
    }

    /// Close the active batch and promote its edits to a single
    /// [`EditAction`] on the undo stack.
    ///
    /// Returns the committed (possibly truncated to
    /// [`MAX_EDITS_PER_ACTION`]) action, or `None` if no batch was open or
    /// the batch was empty. The returned `EditAction` is identical in
    /// length to the one that landed on the undo stack — important for
    /// callers (AI bridge, CLI) that report "committed N edits".
    pub fn commit_batch(&mut self) -> Option<EditAction> {
        let batch = self.active_batch.take()?;
        if batch.edits.is_empty() {
            return None;
        }
        let action = EditAction { edits: batch.edits };
        Some(self.push_truncated(action))
    }

    /// Discard the active batch without recording an undo entry. Returns the
    /// dropped batch so callers can log how many edits were rolled back.
    pub fn abort_batch(&mut self) -> Option<EditBatch> {
        self.active_batch.take()
    }

    /// Push a single [`BlockEdit`] into the active batch. Returns `true` if
    /// the edit landed in a batch, `false` if no batch is open. In practice
    /// all production call-sites discard the return value; it exists so unit
    /// tests can assert batching behaviour.
    pub fn push_edit_batched(&mut self, edit: BlockEdit) -> bool {
        if let Some(batch) = self.active_batch.as_mut() {
            batch.edits.push(edit);
            true
        } else {
            false
        }
    }

    /// `true` while a batch is open.
    pub fn is_batching(&self) -> bool {
        self.active_batch.is_some()
    }

    /// Name of the active batch, if any.
    pub fn batch_name(&self) -> Option<&str> {
        self.active_batch.as_ref().map(|b| b.name.as_str())
    }

    /// Number of edits accumulated in the active batch (0 if none open).
    pub fn batch_size(&self) -> usize {
        self.active_batch
            .as_ref()
            .map(|b| b.edits.len())
            .unwrap_or(0)
    }

    /// Peek at the next undo action without removing it.
    pub fn peek_undo(&self) -> Option<&EditAction> {
        self.undo_stack.front()
    }

    /// Peek at the next redo action without removing it.
    pub fn peek_redo(&self) -> Option<&EditAction> {
        self.redo_stack.front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(x: i32, y: i32, z: i32, old: u16, new: u16) -> BlockEdit {
        BlockEdit {
            x,
            y,
            z,
            old_block: old,
            new_block: new,
        }
    }

    #[test]
    fn push_and_pop_undo() {
        let mut state = UndoRedoState::default();
        state.push(EditAction {
            edits: vec![edit(0, 0, 0, 1, 0), edit(1, 0, 0, 2, 0)],
        });
        assert!(state.can_undo());
        assert!(!state.can_redo());
        let action = state.pop_undo().unwrap();
        assert_eq!(action.edits.len(), 2);
        assert!(!state.can_undo());
        assert!(state.can_redo());
    }

    #[test]
    fn pop_redo_restores_undo() {
        let mut state = UndoRedoState::default();
        state.push(EditAction {
            edits: vec![edit(0, 0, 0, 1, 0)],
        });
        state.pop_undo();
        let action = state.pop_redo().unwrap();
        assert_eq!(action.edits[0].old_block, 1);
        assert!(state.can_undo());
        assert!(!state.can_redo());
    }

    #[test]
    fn new_push_clears_redo() {
        let mut state = UndoRedoState::default();
        state.push(EditAction {
            edits: vec![edit(0, 0, 0, 1, 0)],
        });
        state.pop_undo();
        assert!(state.can_redo());
        state.push(EditAction {
            edits: vec![edit(1, 0, 0, 2, 0)],
        });
        assert!(!state.can_redo());
    }

    #[test]
    fn empty_action_ignored() {
        let mut state = UndoRedoState::default();
        state.push(EditAction { edits: vec![] });
        assert!(!state.can_undo());
    }

    #[test]
    fn undo_limit() {
        let mut state = UndoRedoState::default();
        for i in 0..150 {
            state.push(EditAction {
                edits: vec![edit(i, 0, 0, 1, 0)],
            });
        }
        assert!(state.undo_count() <= MAX_UNDO);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut state = UndoRedoState::default();
        assert!(state.pop_undo().is_none());
        assert!(state.pop_redo().is_none());
    }

    // --- Batch transaction tests -----------------------------------------

    #[test]
    fn begin_and_commit_batch_single_undo_entry() {
        let mut state = UndoRedoState::default();
        assert!(state.begin_batch("fill aabb"));
        assert!(state.is_batching());
        assert_eq!(state.batch_name(), Some("fill aabb"));

        for i in 0..5 {
            assert!(state.push_edit_batched(edit(i, 0, 0, 0, 1)));
        }
        assert_eq!(state.batch_size(), 5);

        let committed = state.commit_batch().expect("committed action");
        assert_eq!(committed.edits.len(), 5);
        assert!(!state.is_batching());

        // The whole batch collapses to a single undo entry.
        assert_eq!(state.undo_count(), 1);
        let popped = state.pop_undo().unwrap();
        assert_eq!(popped.edits.len(), 5);
    }

    #[test]
    fn commit_empty_batch_yields_no_undo_entry() {
        let mut state = UndoRedoState::default();
        assert!(state.begin_batch("noop"));
        assert!(state.commit_batch().is_none());
        assert!(!state.can_undo());
    }

    #[test]
    fn abort_batch_discards_edits_without_undo_entry() {
        let mut state = UndoRedoState::default();
        state.begin_batch("throwaway");
        for i in 0..50 {
            state.push_edit_batched(edit(i, 0, 0, 0, 1));
        }
        let aborted = state.abort_batch().expect("aborted batch");
        assert_eq!(aborted.name, "throwaway");
        assert_eq!(aborted.edits.len(), 50);
        assert!(!state.is_batching());
        assert!(!state.can_undo(), "abort must not record an undo entry");
    }

    #[test]
    fn abort_after_abort_returns_none() {
        let mut state = UndoRedoState::default();
        state.begin_batch("x");
        assert!(state.abort_batch().is_some());
        // Second abort: no batch open, so None.
        assert!(state.abort_batch().is_none());
        assert!(!state.is_batching());
    }

    #[test]
    fn double_begin_returns_false_and_keeps_old_batch() {
        let mut state = UndoRedoState::default();
        assert!(state.begin_batch("outer"));
        // First edit goes to "outer".
        state.push_edit_batched(edit(0, 0, 0, 0, 1));
        // Second begin fails — outer batch untouched.
        assert!(!state.begin_batch("inner"));
        assert_eq!(state.batch_name(), Some("outer"));
        assert_eq!(state.batch_size(), 1);
        // Drop it cleanly so the test doesn't leave a dangling active_batch
        // (the Drop behaviour is no-op, but explicit is clearer).
        state.abort_batch();
    }

    #[test]
    fn push_edit_batched_without_batch_returns_false() {
        let mut state = UndoRedoState::default();
        assert!(!state.push_edit_batched(edit(0, 0, 0, 0, 1)));
        assert!(!state.can_undo());
    }

    #[test]
    fn batch_then_commit_then_new_push_clears_redo() {
        let mut state = UndoRedoState::default();
        state.begin_batch("a");
        state.push_edit_batched(edit(1, 0, 0, 0, 1));
        let act = state.commit_batch().unwrap();
        let _ = state.pop_undo().expect("undo act");
        assert!(state.can_redo());

        // A new batch commits and clears the redo stack, like a regular push.
        state.begin_batch("b");
        state.push_edit_batched(edit(2, 0, 0, 0, 1));
        state.commit_batch();
        assert!(!state.can_redo());

        // The committed action shape is preserved.
        assert_eq!(act.edits.len(), 1);
    }

    #[test]
    fn batch_respects_max_edits_per_action() {
        let mut state = UndoRedoState::default();
        state.begin_batch("huge");
        // Push past the truncation threshold; commit should still work and
        // land a truncated entry on the undo stack.
        for i in 0..(MAX_EDITS_PER_ACTION + 1000) {
            state.push_edit_batched(edit(i as i32, 0, 0, 0, 1));
        }
        let committed = state.commit_batch().unwrap();
        // The returned action AND the action on the undo stack are both
        // truncated to MAX_EDITS_PER_ACTION so callers see a consistent
        // committed-count regardless of size.
        assert_eq!(committed.edits.len(), MAX_EDITS_PER_ACTION);
        assert_eq!(state.undo_count(), 1);
        let from_stack = state.pop_undo().unwrap();
        assert_eq!(from_stack.edits.len(), MAX_EDITS_PER_ACTION);
    }
}
