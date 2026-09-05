/// Resource: which view mode the player is in.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ViewMode {
    #[default]
    FirstPerson,
    ThirdPerson {
        distance: f32,
        angle: f32,
    },
}
