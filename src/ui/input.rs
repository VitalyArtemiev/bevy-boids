//! Player input: one enhanced-input context ([`PlayerContext`]) owning every
//! gameplay action, their default bindings, persistence of overrides
//! (bevy-settings), and the key-binding UI used for remapping.
//!
//! Gameplay systems never read `ButtonInput` for actions — they poll the
//! action entities via [`ActionTag`] + [`TriggerState`]/[`ActionEvents`]
//! (helpers: [`started`], [`fired`], [`completed`]). Raw input is read in
//! exactly two places: the rebind capture and the camera plugin's own fields.

use bevy::prelude::*;
use bevy::settings::{ReflectSettingsGroup, SaveSettingsDeferred, SettingsGroup};
use bevy_egui::egui;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};
use bevy_enhanced_input::prelude::*;
// Not re-exported at the crate root; re-exported here so the rest of the
// codebase never touches `bevy_enhanced_input` paths directly.
pub use bevy_enhanced_input::prelude::{ActionEvents, Binding, Bindings, TriggerState};

/// Marker component for the single player input context entity.
#[derive(Component)]
pub struct PlayerContext;

// --- Actions ---------------------------------------------------------------

macro_rules! bool_action {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(InputAction)]
            #[action_output(bool)]
            pub struct $name;
        )*
    };
}

bool_action!(
    Select,         // LMB drag-select
    SelectAdditive, // Shift held while selecting
    Frontage,       // RMB drag frontage designation
    AdjustWidth,    // Ctrl held while designating (fit columns to width)
    Pause,          // Esc
    ToggleDebug,    // F1
    ToggleTerrain,  // F3
    ToggleScenes,   // F4
);

// Quick-group slots: plain digit recalls the group, Ctrl+digit assigns.
#[derive(InputAction)]
#[action_output(bool)]
pub struct RecallGroup1;
#[derive(InputAction)]
#[action_output(bool)]
pub struct AssignGroup1;
#[derive(InputAction)]
#[action_output(bool)]
pub struct RecallGroup2;
#[derive(InputAction)]
#[action_output(bool)]
pub struct AssignGroup2;
#[derive(InputAction)]
#[action_output(bool)]
pub struct RecallGroup3;
#[derive(InputAction)]
#[action_output(bool)]
pub struct AssignGroup3;
#[derive(InputAction)]
#[action_output(bool)]
pub struct RecallGroup4;
#[derive(InputAction)]
#[action_output(bool)]
pub struct AssignGroup4;
#[derive(InputAction)]
#[action_output(bool)]
pub struct RecallGroup5;
#[derive(InputAction)]
#[action_output(bool)]
pub struct AssignGroup5;
#[derive(InputAction)]
#[action_output(bool)]
pub struct RecallGroup6;
#[derive(InputAction)]
#[action_output(bool)]
pub struct AssignGroup6;

// --- Action identity + defaults --------------------------------------------

/// Stable identity of every rebindable action. Doubles as the persistence
/// key (`BindingsSettings`) and the egui label source.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionId {
    Select,
    SelectAdditive,
    Frontage,
    AdjustWidth,
    Pause,
    ToggleDebug,
    ToggleTerrain,
    ToggleScenes,
    RecallGroup(u8),
    AssignGroup(u8),
}

impl ActionId {
    pub const ALL: [ActionId; 20] = [
        ActionId::Select,
        ActionId::SelectAdditive,
        ActionId::Frontage,
        ActionId::AdjustWidth,
        ActionId::Pause,
        ActionId::ToggleDebug,
        ActionId::ToggleTerrain,
        ActionId::ToggleScenes,
        ActionId::RecallGroup(1),
        ActionId::RecallGroup(2),
        ActionId::RecallGroup(3),
        ActionId::RecallGroup(4),
        ActionId::RecallGroup(5),
        ActionId::RecallGroup(6),
        ActionId::AssignGroup(1),
        ActionId::AssignGroup(2),
        ActionId::AssignGroup(3),
        ActionId::AssignGroup(4),
        ActionId::AssignGroup(5),
        ActionId::AssignGroup(6),
    ];

    pub fn label(&self) -> String {
        match self {
            ActionId::Select => "Select".into(),
            ActionId::SelectAdditive => "Add to selection".into(),
            ActionId::Frontage => "Designate frontage".into(),
            ActionId::AdjustWidth => "Fit columns to width".into(),
            ActionId::Pause => "Pause".into(),
            ActionId::ToggleDebug => "Toggle debug panel".into(),
            ActionId::ToggleTerrain => "Toggle terrain panel".into(),
            ActionId::ToggleScenes => "Toggle scene menu".into(),
            ActionId::RecallGroup(n) => format!("Recall group {n}"),
            ActionId::AssignGroup(n) => format!("Assign group {n}"),
        }
    }

    /// Default binding. Every action has exactly one; recall/assign group
    /// share a digit, split by the Ctrl modifier.
    fn default_binding(&self) -> Binding {
        match self {
            ActionId::Select => MouseButton::Left.into(),
            ActionId::SelectAdditive => KeyCode::ShiftLeft.into(),
            ActionId::Frontage => MouseButton::Right.into(),
            ActionId::AdjustWidth => KeyCode::ControlLeft.into(),
            ActionId::Pause => KeyCode::Escape.into(),
            ActionId::ToggleDebug => KeyCode::F1.into(),
            ActionId::ToggleTerrain => KeyCode::F3.into(),
            ActionId::ToggleScenes => KeyCode::F4.into(),
            ActionId::RecallGroup(n) => Binding::Keyboard {
                key: digit(*n),
                mod_keys: ModKeys::empty(),
            },
            ActionId::AssignGroup(n) => Binding::Keyboard {
                key: digit(*n),
                mod_keys: ModKeys::CONTROL,
            },
        }
    }
}

fn digit(n: u8) -> KeyCode {
    match n {
        1 => KeyCode::Digit1,
        2 => KeyCode::Digit2,
        3 => KeyCode::Digit3,
        4 => KeyCode::Digit4,
        5 => KeyCode::Digit5,
        _ => KeyCode::Digit6,
    }
}

/// Tags action entities with their stable identity so systems and the
/// binding UI can enumerate actions without per-type queries.
#[derive(Component, Debug, Clone, Copy)]
pub struct ActionTag(pub ActionId);

// --- Persistence -----------------------------------------------------------

/// A single rebindable input source.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RebindableInput {
    Keyboard(KeyCode),
    Mouse(MouseButton),
}

impl From<RebindableInput> for Binding {
    fn from(input: RebindableInput) -> Self {
        match input {
            RebindableInput::Keyboard(key) => key.into(),
            RebindableInput::Mouse(button) => button.into(),
        }
    }
}

impl std::fmt::Display for RebindableInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebindableInput::Keyboard(key) => write!(f, "{key:?}"),
            RebindableInput::Mouse(button) => write!(f, "{button:?}"),
        }
    }
}

/// The input source of a binding without modifier context, for comparing
/// against a captured [`RebindableInput`].
fn binding_source(binding: &Binding) -> Option<RebindableInput> {
    match *binding {
        Binding::Keyboard { key, .. } => Some(RebindableInput::Keyboard(key)),
        Binding::MouseButton { button, .. } => Some(RebindableInput::Mouse(button)),
        _ => None,
    }
}

fn same_source(binding: &Binding, input: RebindableInput) -> bool {
    binding_source(binding) == Some(input)
}

/// Persisted key overrides, one entry per action whose binding differs
/// from its default. Loaded by bevy-settings at startup.
#[derive(Resource, Reflect, SettingsGroup, Debug, Default, Clone, PartialEq)]
#[reflect(Default, Resource)]
pub struct BindingsSettings {
    pub overrides: Vec<BindingOverride>,
}

#[derive(Reflect, Debug, Clone, Copy, PartialEq)]
pub struct BindingOverride {
    pub action: ActionId,
    pub input: RebindableInput,
}

// --- Plugin ----------------------------------------------------------------

pub struct InputPlugin;

/// Whether the key-bindings window is open (toggled from the Options
/// window).
#[derive(Resource, Default)]
pub struct BindingsOpen(pub bool);

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        // Must run before UiPlugin adds SettingsPlugin: the settings loader
        // scans the type registry at plugin build time.
        app.register_type::<BindingsSettings>()
            .register_type::<BindingOverride>()
            .register_type::<ActionId>()
            .register_type::<RebindableInput>()
            .register_type::<KeyCode>()
            .register_type::<MouseButton>()
            .register_type_data::<BindingsSettings, ReflectSettingsGroup>()
            .add_plugins(EnhancedInputPlugin)
            .add_input_context::<PlayerContext>()
            .init_resource::<BindingsOpen>()
            .add_systems(Startup, spawn_input_context)
            .add_systems(
                EguiPrimaryContextPass,
                bindings_ui.run_if(
                    in_state(crate::ui::GameState::MainMenu)
                        .or_else(in_state(crate::ui::GameState::Paused)),
                ),
            );
    }
}

/// Spawns the context entity with all actions at their (persisted)
/// bindings. Runs after bevy-settings has inserted `BindingsSettings`.
fn spawn_input_context(mut commands: Commands, settings: Res<BindingsSettings>) {
    let overrides: Vec<(ActionId, RebindableInput)> = settings
        .overrides
        .iter()
        .map(|o| (o.action, o.input))
        .collect();
    let override_for = move |id: ActionId| -> Option<RebindableInput> {
        overrides.iter().find(|(a, _)| *a == id).map(|(_, i)| *i)
    };

    commands.spawn((
        PlayerContext,
        actions!(
            PlayerContext[
                (ActionTag(ActionId::Select), Action::<Select>::new(), bindings![Binding::from(MouseButton::Left)]),
                (ActionTag(ActionId::SelectAdditive), Action::<SelectAdditive>::new(), bindings![Binding::from(KeyCode::ShiftLeft)]),
                (ActionTag(ActionId::Frontage), Action::<Frontage>::new(), bindings![Binding::from(MouseButton::Right)]),
                (ActionTag(ActionId::AdjustWidth), Action::<AdjustWidth>::new(), bindings![Binding::from(KeyCode::ControlLeft)]),
                (ActionTag(ActionId::Pause), Action::<Pause>::new(), bindings![Binding::from(KeyCode::Escape)]),
                (ActionTag(ActionId::ToggleDebug), Action::<ToggleDebug>::new(), bindings![Binding::from(KeyCode::F1)]),
                (ActionTag(ActionId::ToggleTerrain), Action::<ToggleTerrain>::new(), bindings![Binding::from(KeyCode::F3)]),
                (ActionTag(ActionId::ToggleScenes), Action::<ToggleScenes>::new(), bindings![Binding::from(KeyCode::F4)]),
                (ActionTag(ActionId::RecallGroup(1)), Action::<RecallGroup1>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit1, mod_keys: ModKeys::empty() }]),
                (ActionTag(ActionId::RecallGroup(2)), Action::<RecallGroup2>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit2, mod_keys: ModKeys::empty() }]),
                (ActionTag(ActionId::RecallGroup(3)), Action::<RecallGroup3>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit3, mod_keys: ModKeys::empty() }]),
                (ActionTag(ActionId::RecallGroup(4)), Action::<RecallGroup4>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit4, mod_keys: ModKeys::empty() }]),
                (ActionTag(ActionId::RecallGroup(5)), Action::<RecallGroup5>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit5, mod_keys: ModKeys::empty() }]),
                (ActionTag(ActionId::RecallGroup(6)), Action::<RecallGroup6>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit6, mod_keys: ModKeys::empty() }]),
                (ActionTag(ActionId::AssignGroup(1)), Action::<AssignGroup1>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit1, mod_keys: ModKeys::CONTROL }]),
                (ActionTag(ActionId::AssignGroup(2)), Action::<AssignGroup2>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit2, mod_keys: ModKeys::CONTROL }]),
                (ActionTag(ActionId::AssignGroup(3)), Action::<AssignGroup3>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit3, mod_keys: ModKeys::CONTROL }]),
                (ActionTag(ActionId::AssignGroup(4)), Action::<AssignGroup4>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit4, mod_keys: ModKeys::CONTROL }]),
                (ActionTag(ActionId::AssignGroup(5)), Action::<AssignGroup5>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit5, mod_keys: ModKeys::CONTROL }]),
                (ActionTag(ActionId::AssignGroup(6)), Action::<AssignGroup6>::new(), bindings![Binding::Keyboard { key: KeyCode::Digit6, mod_keys: ModKeys::CONTROL }]),
            ]
        ),
    ));
    // Deferred: the action entities exist only once commands apply, so the
    // override pass runs after the spawn on the same frame's flush.
    commands.queue(move |world: &mut World| {
        let tagged: Vec<(Entity, ActionId)> = world
            .query::<(Entity, &ActionTag)>()
            .iter(world)
            .map(|(entity, tag)| (entity, tag.0))
            .collect();
        for (entity, tag) in tagged {
            if let Some(input) = override_for(tag) {
                let current = world
                    .get::<Bindings>(entity)
                    .and_then(|b| b.iter().next())
                    .and_then(|b| world.get::<Binding>(b).copied());
                if let Some(current) = current {
                    replace_binding(world, tag, keep_modifiers(current, input));
                }
            }
        }
    });
}

/// Rebinds `action` to `input`, keeping the action's modifier requirement
/// (e.g. Ctrl on assign-group). If another action already uses `input`, the
/// two swap bindings. Updates `BindingsSettings` and saves.
pub fn rebind(world: &mut World, action: ActionId, input: RebindableInput) {
    let mut current: Vec<(ActionId, Binding)> = Vec::new();
    for (_, tag, bindings) in world.query::<(Entity, &ActionTag, &Bindings)>().iter(world) {
        if let Some(binding) = bindings.iter().next().and_then(|b| world.get::<Binding>(b)) {
            current.push((tag.0, *binding));
        }
    }

    let Some(&(_, action_binding)) = current.iter().find(|(id, _)| *id == action) else {
        return;
    };
    let new_binding = keep_modifiers(action_binding, input);

    // Conflict: the other action that already uses this input gets our old
    // binding (swap).
    if let Some(&(other_id, _)) = current
        .iter()
        .find(|(id, binding)| *id != action && same_source(binding, input))
    {
        replace_binding(world, other_id, action_binding);
    }
    replace_binding(world, action, new_binding);
}

/// Swaps an action's binding entity for one carrying `binding` (Binding is
/// immutable, so this is despawn + respawn), records the override in
/// `BindingsSettings`, and saves.
fn replace_binding(world: &mut World, action: ActionId, binding: Binding) {
    let action_entity = world
        .query::<(Entity, &ActionTag)>()
        .iter(world)
        .find(|(_, tag)| tag.0 == action)
        .map(|(entity, _)| entity)
        // End the query borrow before mutating.
        ;
    let Some(action_entity) = action_entity else {
        return;
    };
    world
        .entity_mut(action_entity)
        .despawn_related::<Bindings>()
        .insert(bindings![binding]);

    // Persist: record (or drop, when back at the default) the override.
    let source = binding_source(&binding).unwrap();
    let is_default = binding_source(&action.default_binding()) == Some(source);
    let mut settings = world.resource_mut::<BindingsSettings>();
    settings.overrides.retain(|o| o.action != action);
    if !is_default {
        settings.overrides.push(BindingOverride {
            action,
            input: source,
        });
    }
    drop(settings);
    let mut queue = bevy::ecs::world::CommandQueue::default();
    queue.push(SaveSettingsDeferred::default());
    queue.apply(world);
}

/// Builds a binding for `input` that preserves the modifier requirement of
/// `old` (Ctrl for assign-group, none otherwise).
fn keep_modifiers(old: Binding, input: RebindableInput) -> Binding {
    let mod_keys = match old {
        Binding::Keyboard { mod_keys, .. } | Binding::MouseButton { mod_keys, .. } => mod_keys,
        _ => ModKeys::empty(),
    };
    match input {
        RebindableInput::Keyboard(key) => Binding::Keyboard { key, mod_keys },
        RebindableInput::Mouse(button) => Binding::MouseButton { button, mod_keys },
    }
}

// --- Polling helpers -------------------------------------------------------

/// True on the frame the action started triggering (≈ just pressed).
pub fn started(actions: &Query<(&ActionTag, &TriggerState, &ActionEvents)>, id: ActionId) -> bool {
    actions
        .iter()
        .any(|(tag, _, events)| tag.0 == id && events.contains(ActionEvents::START))
}

/// True while the action triggers (≈ held).
pub fn fired(actions: &Query<(&ActionTag, &TriggerState, &ActionEvents)>, id: ActionId) -> bool {
    actions
        .iter()
        .any(|(tag, state, _)| tag.0 == id && *state == TriggerState::Fired)
}

/// True on the frame the action stopped triggering (≈ just released).
pub fn completed(
    actions: &Query<(&ActionTag, &TriggerState, &ActionEvents)>,
    id: ActionId,
) -> bool {
    actions
        .iter()
        .any(|(tag, _, events)| tag.0 == id && events.contains(ActionEvents::COMPLETE))
}

// --- Key-bindings UI --------------------------------------------------------

/// The action's current binding: its first bound input, or the default when
/// the input context hasn't spawned. Used by the rebind list here and by the
/// Help panel so both stay accurate across remaps.
pub fn current_binding(
    q_actions: &Query<(&ActionTag, &Bindings)>,
    q_bindings: &Query<&Binding>,
    id: ActionId,
) -> Binding {
    q_actions
        .iter()
        .find(|(tag, _)| tag.0 == id)
        .and_then(|(_, bindings)| bindings.iter().next())
        .and_then(|e| q_bindings.get(e).ok().copied())
        .unwrap_or_else(|| id.default_binding())
}

/// The key-bindings window: one row per action with its current binding,
/// a rebind button that captures the next key/mouse press, and a
/// reset-to-defaults button. Rebinds that collide with another action
/// swap the two bindings.
fn bindings_ui(
    mut contexts: EguiContexts,
    mut open: ResMut<BindingsOpen>,
    mut capturing: Local<Option<ActionId>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    q_actions: Query<(&ActionTag, &Bindings)>,
    q_bindings: Query<&Binding>,
    mut commands: Commands,
) -> Result {
    // Capture runs before the early return so a press resolves even on the
    // frame the window is closed.
    if let Some(action) = *capturing {
        let pressed: Option<RebindableInput> = keys
            .get_just_pressed()
            .next()
            .map(|k| RebindableInput::Keyboard(*k))
            .or_else(|| {
                mouse
                    .get_just_pressed()
                    .next()
                    .map(|b| RebindableInput::Mouse(*b))
            });
        if let Some(input) = pressed {
            commands.queue(move |world: &mut World| rebind(world, action, input));
            *capturing = None;
        } else if mouse.just_released(MouseButton::Right) {
            // Releasing the button that opened the window shouldn't be
            // captured as a binding the moment it closes.
        }
    }
    if !open.0 {
        *capturing = None;
        return Ok(());
    }

    let ctx = contexts.ctx_mut()?;
    let mut reset_all = false;
    egui::Window::new("Key Bindings")
        .open(&mut open.0)
        .min_width(340.0)
        .show(ctx, |ui| {
            egui::Grid::new("bindings")
                .num_columns(3)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    for id in ActionId::ALL {
                        ui.label(id.label());
                        // `Binding`'s Display renders modifiers ("Ctrl + Digit1").
                        let text = current_binding(&q_actions, &q_bindings, id).to_string();
                        ui.strong(text);
                        if *capturing == Some(id) {
                            let rebind = ui.button("press a key…").on_hover_text("Click to cancel");
                            if rebind.clicked() {
                                *capturing = None;
                            }
                        } else if ui
                            .button("Rebind")
                            .on_hover_text(
                                "Capture the next key or mouse press; a clash swaps bindings with the other action.",
                            )
                            .clicked()
                        {
                            *capturing = Some(id);
                        }
                        ui.end_row();
                    }
                });
            ui.add_space(6.0);
            if ui
                .button("Reset all to defaults")
                .on_hover_text("Restore every binding to its default.")
                .clicked()
            {
                reset_all = true;
            }
        });
    if reset_all {
        *capturing = None;
        commands.queue(|world: &mut World| {
            for id in ActionId::ALL {
                replace_binding(world, id, id.default_binding());
            }
        });
    }
    Ok(())
}
