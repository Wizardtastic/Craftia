/// Resource: which view mode the player is in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewMode {
    FirstPerson,
    ThirdPerson { distance: f32, angle: f32 },
}

impl Default for ViewMode {
    fn default() -> Self {
        ViewMode::FirstPerson
    }
}
