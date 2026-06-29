import pyvista as pv

from oil_gestures.simulator.details_and_particles import Body, Valve, Manometer, Plug


Details = {
    1 :  ("Main_body", Body, "white", [11]),
    2 :  ("connections", Body, "darkred", [0]),
    3 :  ("connections1", Body, "silver", [1]),
    4 :  ("connections2", Body, "white", [10]),
    5 :  ("valve_1", Valve, "red", [2]),
    6 :  ("valve_2", Valve, "red", [3]),
    7 :  ("valve_3", Valve, "red", [4]),
    8 :  ("valve_4", Valve, "red", [5]),
    9 :  ("valve_5", Valve, "red", [6]),
    10 : ("valve_11", Valve, "red", [8]),
    11 : ("valve_12", Valve, "red", [12]),
    12 : ("valve_13", Valve, "red", [14]),
    13 : ("valve_14", Valve, "red", [16]),
    14 : ("valve_15", Valve, "red", [18]),
    15 : ("manometer_1", Manometer, "black", [9]),
    16 : ("manometer_2", Manometer, "black", [13]),
    17 : ("manometer_3", Manometer, "black", [15]),
    18 : ("manometer_4", Manometer, "black", [17]),
    19 : ("plug", Plug, "red", [7]),
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


def load_model(plotter, filepath):
    print("  Загрузка модели ...")
    blocks = pv.read(filepath)
    parts = extract_all_meshes(blocks)
    model = parts[0].merge(parts[1:])
    bounds = model.bounds
    offset = [
        -(bounds[0] + bounds[1]) / 2,
        -bounds[2],
        -(bounds[4] + bounds[5]) / 2,
    ]
    details = []
    for key in sorted(Details.keys()):
        name, cls, color, indices = Details[key]
        elements = [parts[i].copy() for i in indices]
        if len(elements) > 1:
            obj = elements[0].merge(elements[1:])
        else:
            obj = elements[0]
        obj.translate(offset, inplace=True)
        actor = plotter.add_mesh(obj, color=color, show_edges=False)
        color = actor.GetProperty().GetColor()
        detail = cls(obj, actor, name, color)
        details.append(detail)


    blocks1 = pv.read("assets/controller.glb")
    parts1 = extract_all_meshes(blocks1)
    print(len(parts1))
    obj = parts1[21]
    offset[0] += 8
    offset[1] -= 1.87
    obj.translate(offset, inplace=True)
    center = obj.center
    obj.rotate_vector((0, 1, 0), -31, point=center, inplace=True)
    actor = plotter.add_mesh(obj, color="silver", show_edges=False)
    color = actor.GetProperty().GetColor()
    detail = Body(obj, actor, "controller", color)
    details.append(detail)

    blocks1 = pv.read("assets/level_gauge.glb")
    parts1 = extract_all_meshes(blocks1)
    print(len(parts1))
    obj = parts1[1]
    offset[0] -= 14
    offset[1] += 0.78
    offset[2] -= 0.36
    obj.translate(offset, inplace=True)
    center = obj.center
    actor = plotter.add_mesh(obj, color="silver", show_edges=False)
    color = actor.GetProperty().GetColor()
    detail = Body(obj, actor, "level_gauge", color)
    details.append(detail)


    print("  Модель успешно загружена")
    return details