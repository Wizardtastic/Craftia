//! Property animation system: evaluates keyframes and applies to entity properties.

use voxel_animation::data::TargetedProperty;
use voxel_ecs::World;

use crate::components::property_animator::{PropertyAnimator, PropertyTarget};
use crate::components::Transform;

/// Property animation system entry point. Called each fixed timestep.
pub fn property_animation_system(world: &mut World, dt: f32) {
    // Collect entities with PropertyAnimator to avoid borrow issues.
    let entities: Vec<voxel_ecs::Entity> =
        world.query::<&PropertyAnimator>().map(|(e, _)| e).collect();

    for entity in entities {
        // Tick the animator.
        let Some(mut animator) = world.get::<PropertyAnimator>(entity).cloned() else {
            continue;
        };

        if !animator.playing || animator.clip.is_none() {
            continue;
        }

        animator.tick(dt);

        // Sample the clip at current time.
        let clip = animator.clip.as_ref().unwrap();
        let time = animator.time;

        // Find the channel targeting Translation (for Position target).
        for channel in &clip.channels {
            let should_apply = matches!(
                (&animator.target, &channel.property),
                (PropertyTarget::Position, TargetedProperty::Translation)
                    | (PropertyTarget::RotationEuler, TargetedProperty::Rotation)
                    | (PropertyTarget::Scale, TargetedProperty::Scale)
            );

            if !should_apply {
                continue;
            }

            let value = voxel_animation::sampling::sample_channel(channel, time);

            // Apply to the entity's Transform.
            if let Some(transform) = world.get_mut::<Transform>(entity) {
                match &animator.target {
                    PropertyTarget::Position => {
                        transform.pos = glam::Vec3::new(value[0], value[1], value[2]);
                    }
                    PropertyTarget::RotationEuler => {
                        let euler = glam::EulerRot::YXZ;
                        transform.rot = glam::Quat::from_euler(
                            euler,
                            value[0].to_radians(),
                            value[1].to_radians(),
                            value[2].to_radians(),
                        );
                    }
                    PropertyTarget::Scale => {
                        // Scale is not directly on Transform, but we could add it.
                        // For now, this is a no-op.
                    }
                }
            }
        }

        // Write back the updated animator.
        world.set(entity, animator);
    }
}
