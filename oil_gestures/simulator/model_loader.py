import pyvista as pv

from oil_gestures.simulator.details_and_particles import Body, Valve, Manometer, Plug, Flap, LevelGaugeAssembly, LevelGaugeScreen, ParticleSystem, ControllerDoor, ControllerScreen, ControllerLever

DETAILS = {
    1: ("main_body", Body, "white", [11], (0, 1, 0), 1.0),
    2: ("connections", Body, "darkred", [0], (0, 1, 0), 1.0),
    3: ("connections1", Body, "silver", [1], (0, 1, 0), 1.0),
    4: ("connections2", Body, "white", [10], (0, 1, 0), 1.0),
    5: ("valve_1", Valve, "red", [2], (0, 0, 1), 1.0),
    6: ("valve_2", Valve, "red", [3], (0, 0, 1), 1.0),
    7: ("valve_3", Valve, "red", [4], (0, 0, 1), 1.0),
    8: ("valve_4", Valve, "red", [5], (0, 0, 1), 1.0),
    9: ("valve_5", Valve, "red", [6], (0, 0, 1), 1.0),
    10: ("valve_11", Valve, "red", [8], (0, 0, 1), 1.1),
    11: ("valve_12", Valve, "red", [12], (0, 0, 1), 1.1),
    12: ("valve_13", Valve, "red", [14], (0, 0, 1), 1.1),
    13: ("valve_14", Valve, "red", [16], (0, 0, 1), 1.1),
    14: ("valve_15", Valve, "red", [18], (0, 1, 0), 1.3),
    15: ("manometer_1", Manometer, "black", [9], (0, 0, 1), 1.4),
    16: ("manometer_2", Manometer, "black", [13], (0, 0, 1), 1.4),
    17: ("manometer_3", Manometer, "black", [15], (0, 0, 1), 1.4),
    18: ("manometer_4", Manometer, "black", [17], (0, 0, 1), 1.4),
    19: ("plug", Plug, "red", [7], (1, 0, 0), 1),
}

CONTROLLER_CONFIG = {
    "file": "assets/controller.glb",
    "offset": (8.0, -1, 0.0),
    "rotation_y": 0.0,
    "parts": {
        0: ("controller_door", "#8a7a45"),
        1: ("controller_screen", "#1d2a1d"),
        2: ("controller_panel", "#3a3a3a"),
        3: ("controller_circle_one", "silver"),
        4: ("controller_circle_two", "silver"),
        5: ("controller_circle_three", "silver"),
        6: ("controller_circle_four", "silver"),
        7: ("controller_circle_five", "silver"),
        8: ("controller_start_button", "green"),
        9: ("controller_stop_button", "red"),
        10: ("controller_button_one", "silver"),
        11: ("controller_button_two", "silver"),
        12: ("controller_button_lower", "silver"),
        13: ("controller_button_top", "silver"),
        14: ("controller_button_left", "silver"),
        15: ("controller_button_center", "silver"),
        16: ("controller_button_right", "silver"),
        17: ("controller_button_long", "silver"),
        18: ("controller_black_wheel", "black"),
        19: ("controller_big_button", "yellow"),
        20: ("controller_base_small", "darkgray"),
        21: ("controller_body", "silver"),
        22: ("controller_lever_on_off", "darkgray"),
    },
}

LEVEL_GAUGE_CONFIG = {
    "file": "assets/level_gauge.glb",
    "scale": 0.44,
    "rotation_y": 0.0,
    "mount_to": "plug",
    "mount_gap": 0.00,
    "fine_offset": (0.0, 0.09, 0.13),
    "parts": {
    0: ("level_gauge_cover", "#c8ba8b"),
    1: ("level_gauge_base", "#b7a66a"),
    2: ("level_gauge_flap", "#8a7a45"),
    3: ("level_gauge_screen", "#5f615e"),
    4: ("level_gauge_button_mode", "#d6d4ce"),
    5: ("level_gauge_button_input_output", "#969590"),
    6: ("level_gauge_button_level", "silver"),
    7: ("level_gauge_button_return", "#75726a"),
}
}


def extract_all_meshes(multiblock):
    meshes = []
    for i in range(multiblock.n_blocks):
        block = multiblock[i]
        if isinstance(block, pv.MultiBlock):
            meshes.extend(extract_all_meshes(block))
        else:
            meshes.append(block)
    return meshes


def find_detail(details, name):
    for detail in details:
        if detail.name == name:
            return detail
    return None


def add_detail(plotter, details, mesh, cls, name, color, axis=(0, 1, 0)):
    actor = plotter.add_mesh(mesh, color=color, show_edges=False)
    actor_color = actor.GetProperty().GetColor()
    detail = cls(mesh, actor, name, axis, actor_color)
    details.append(detail)
    return detail


def build_base_offset(parts):
    model = parts[0].merge(parts[1:])
    bounds = model.bounds
    return [
        -(bounds[0] + bounds[1]) / 2,
        -bounds[2],
        -(bounds[4] + bounds[5]) / 2,
    ]


def merge_bounds(meshes):
    merged = meshes[0].copy()
    if len(meshes) > 1:
        merged = merged.merge([m.copy() for m in meshes[1:]])
    return merged.bounds


def bounds_center(bounds):
    return (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )


def load_main_installation(plotter, filepath, details):
    blocks = pv.read(filepath)
    parts = extract_all_meshes(blocks)
    base_offset = build_base_offset(parts)

    for key in sorted(DETAILS.keys()):
        name, cls, color, indices, axis, scale = DETAILS[key]
        elements = [parts[i].copy() for i in indices]

        if len(elements) > 1:
            obj = elements[0].merge(elements[1:])
        else:
            obj = elements[0]

        if scale != 1.0:
            center = obj.center
            obj.scale([scale, scale, scale], inplace=True)
            new_center = obj.center
            obj.translate([center[0]-new_center[0], center[1]-new_center[1], center[2]-new_center[2]], inplace=True)
        obj.translate(base_offset, inplace=True)
        add_detail(plotter, details, obj, cls, name, color, axis)

    return base_offset


def load_controller(plotter, details, base_offset):
    blocks = pv.read(CONTROLLER_CONFIG["file"])
    parts = extract_all_meshes(blocks)
    print("controller parts:", len(parts))

    prepared_parts = []

    for index, (name, color) in CONTROLLER_CONFIG["parts"].items():
        mesh = parts[index].copy()
        prepared_parts.append({
            "mesh": mesh,
            "name": name,
            "color": color,
        })

    all_bounds = merge_bounds([p["mesh"] for p in prepared_parts])
    center = bounds_center(all_bounds)

    rotation_y = CONTROLLER_CONFIG["rotation_y"]
    dx, dy, dz = CONTROLLER_CONFIG["offset"]
    controller_offset = [
        base_offset[0] + dx,
        base_offset[1] + dy,
        base_offset[2] + dz,
    ]

    for part in prepared_parts:
        mesh = part["mesh"]
        name = part["name"]
        color = part["color"]

        mesh.rotate_vector((0, 1, 0), rotation_y, point=center, inplace=True)
        mesh.translate(controller_offset, inplace=True)

        actor = plotter.add_mesh(mesh, color=color, show_edges=False)
        actor_color = actor.GetProperty().GetColor()

        if name == "controller_door":
            b = mesh.bounds
            hinge_point = (b[0], (b[2] + b[3]), b[4])

            detail = ControllerDoor(
                mesh,
                actor,
                name,
                actor_color,
                hinge_point=hinge_point,
            )

        elif name == "controller_screen":
            detail = ControllerScreen(mesh, actor, name, plotter, actor_color)

        elif name == "controller_lever_on_off":
            b = mesh.bounds
            pivot_point = (
                (b[0] + b[1]) / 2,
                b[2],
                (b[4] + b[5]) / 2,
            )
            detail = ControllerLever(
                mesh,
                actor,
                name,
                actor_color,
                pivot_point=pivot_point,
            )

        else:
            detail = Body(mesh, actor, name, actor_color)
            detail.highlightable = name not in {
                "controller_body",
                "controller_base_small",
                "controller_screen",
                "controller_panel",
                "controller_circle_one",
                "controller_circle_two",
                "controller_circle_three",
                "controller_circle_four",
                "controller_circle_five",
            }

        details.append(detail)


def load_level_gauge(plotter, details):
    blocks = pv.read(LEVEL_GAUGE_CONFIG["file"])
    parts = extract_all_meshes(blocks)
    print("level_gauge parts:", len(parts))

    meshes = [p.copy() for p in parts]

    scale = LEVEL_GAUGE_CONFIG["scale"]
    for mesh in meshes:
        mesh.scale([scale, scale, scale], inplace=True)

    all_bounds = merge_bounds(meshes)
    center = bounds_center(all_bounds)

    rotation_y = LEVEL_GAUGE_CONFIG["rotation_y"]
    if rotation_y != 0:
        for mesh in meshes:
            mesh.rotate_vector((0, 1, 0), rotation_y, point=center, inplace=True)

    all_bounds = merge_bounds(meshes)

    gauge_mount_point = (
        all_bounds[1],
        (all_bounds[2] + all_bounds[3]) / 2,
        (all_bounds[4] + all_bounds[5]) / 2,
    )

    mount_detail = find_detail(details, LEVEL_GAUGE_CONFIG["mount_to"])
    if mount_detail is None:
        raise RuntimeError(
            f"Не найдена деталь для монтажа уровнемера: {LEVEL_GAUGE_CONFIG['mount_to']}"
        )

    pb = mount_detail.bounds

    target_mount_point = (
        pb[1] + LEVEL_GAUGE_CONFIG["mount_gap"],
        (pb[2] + pb[3]) / 2,
        (pb[4] + pb[5]) / 2,
    )

    fine_dx, fine_dy, fine_dz = LEVEL_GAUGE_CONFIG.get("fine_offset", (0.0, 0.0, 0.0))

    shift = [
        target_mount_point[0] - gauge_mount_point[0] + fine_dx,
        target_mount_point[1] - gauge_mount_point[1] + fine_dy,
        target_mount_point[2] - gauge_mount_point[2] + fine_dz,
    ]

    highlight_parts = {
        "level_gauge_flap",
        "level_gauge_screen",
        "level_gauge_cover",
        "level_gauge_button_input_output",
        "level_gauge_button_level",
        "level_gauge_button_mode",
        "level_gauge_button_return",
    }

    prepared_parts = []

    for i, mesh in enumerate(meshes):
        mesh.translate(shift, inplace=True)

        name, color = LEVEL_GAUGE_CONFIG["parts"].get(
            i,
            (f"level_gauge_part_{i}", "silver")
        )

        prepared_parts.append({
            "index": i,
            "mesh": mesh,
            "name": name,
            "color": color,
            "bounds": mesh.bounds,
        })

    base_part = next((p for p in prepared_parts if p["name"] == "level_gauge_base"), None)
    flap_part = next((p for p in prepared_parts if p["name"] == "level_gauge_flap"), None)

    hinge_point = None

    if base_part is not None and flap_part is not None:
        bb = base_part["bounds"]
        fb = flap_part["bounds"]

        overlap_y_min = max(bb[2], fb[2])
        overlap_y_max = min(bb[3], fb[3])

        overlap_z_min = max(bb[4], fb[4])
        overlap_z_max = min(bb[5], fb[5])

        if overlap_y_min <= overlap_y_max:
            hinge_y = (overlap_y_min + overlap_y_max) / 2
        else:
            hinge_y = (fb[2] + fb[3]) / 2

        if overlap_z_min <= overlap_z_max:
            hinge_z = (overlap_z_min + overlap_z_max) / 2
        else:
            hinge_z = (fb[4] + fb[5]) / 2

        # flap крепится к base со стороны, которая ближе всего к base
        if abs(fb[0] - bb[1]) < abs(fb[1] - bb[0]):
            hinge_x = fb[0]
        else:
            hinge_x = fb[1]

        hinge_point = (hinge_x, hinge_y, hinge_z)
        print("level_gauge flap hinge_point:", hinge_point)
        flap_jet = None
        jet_origin = None
        jet_direction = (-1.0, 0.0, 0.0)

        if hinge_point is not None:
            jet_origin = (
                hinge_point[0] + 0.1,
                hinge_point[1] - 0.08,
                hinge_point[2] - 0.1,
            )

            flap_jet = ParticleSystem(
                plotter,
                position=jet_origin,
                direction=jet_direction,
                particle_type=ParticleSystem.AIR_BLAST,
                count=720,
            )

    level_gauge_parts = []

    for part in prepared_parts:
        mesh = part["mesh"]
        name = part["name"]
        color = part["color"]

        actor = plotter.add_mesh(mesh, color=color, show_edges=False)
        actor_color = actor.GetProperty().GetColor()

        if name == "level_gauge_flap" and hinge_point is not None:
            part_detail = Flap(
                mesh,
                actor,
                name,
                actor_color,
                hinge_point=hinge_point,
                air_jet=flap_jet,
                jet_origin=jet_origin,
                jet_direction=jet_direction,
            )
        elif name == "level_gauge_screen":
            part_detail = LevelGaugeScreen(mesh, actor, name, plotter, actor_color)
        else:
            part_cls = Flap if name == "level_gauge_flap" else Body
            part_detail = part_cls(mesh, actor, name, actor_color)

        part_detail.highlightable = name in highlight_parts

        level_gauge_parts.append(part_detail)
        details.append(part_detail)

    level_gauge = LevelGaugeAssembly("level_gauge", level_gauge_parts)
    details.append(level_gauge)


def load_model(plotter, filepath):
    print("Загрузка модели...")
    details = []

    base_offset = load_main_installation(plotter, filepath, details)
    load_controller(plotter, details, base_offset)
    load_level_gauge(plotter, details)

    print("Модель успешно загружена")
    return details