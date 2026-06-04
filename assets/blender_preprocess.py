import sys
import bpy
import argparse
import mathutils


def recenter_all_meshes():
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    min_coords = mathutils.Vector((9999999, 9999999, 9999999))
    max_coords = mathutils.Vector((-9999999, -9999999, -9999999))
    for obj in (o for o in bpy.context.scene.objects if o.type == 'MESH'):
        for corner in obj.bound_box:
            wc = obj.matrix_world @ mathutils.Vector(corner)
            min_coords.x = min(min_coords.x, wc.x)
            min_coords.y = min(min_coords.y, wc.y)
            min_coords.z = min(min_coords.z, wc.z)
            max_coords.x = max(max_coords.x, wc.x)
            max_coords.y = max(max_coords.y, wc.y)
            max_coords.z = max(max_coords.z, wc.z)

    center = (min_coords + max_coords) / 2.0
    for obj in (o for o in bpy.context.scene.objects if o.type == 'MESH'):
        obj.location -= center


def preprocess_blend(file_path):
    bpy.ops.wm.open_mainfile(filepath=file_path)
    bpy.ops.file.pack_all()
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            for modifier in obj.modifiers:
                if not modifier.show_viewport:
                    modifier.show_viewport = True
                try:
                    bpy.ops.object.modifier_apply(modifier=modifier.name)
                except RuntimeError as e:
                    print(f"Skipping modifier '{modifier.name}' on '{obj.name}': {e}")
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.animation_data:
            obj.animation_data_clear()
            obj.constraints.clear()


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True)
args = parser.parse_args(argv)

if args.file.endswith('.blend'):
    preprocess_blend(args.file)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.file)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

recenter_all_meshes()

out_path = args.file.replace('.blend', '.glb') if args.file.endswith('.blend') else args.file
bpy.ops.export_scene.gltf(
    filepath=out_path,
    export_format='GLB',
    use_visible=True,
    export_apply=True,
    export_materials='EXPORT',
    export_draco_mesh_compression_enable=False,
)
