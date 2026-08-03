/// Resource: which view mode the player is in.
#[derive(Clone, Copy, Debug, PartialEq)]
#[derive(Default)]
pub enum ViewMode {
    #[default]
    FirstPerson,
    ThirdPerson { distance: f32, angle: f32 },
}

