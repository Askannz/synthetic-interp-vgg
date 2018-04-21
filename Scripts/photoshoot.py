import os
import numpy as np
import bpy
from mathutils import Euler, Vector, Color
import json

config = json.load(open('config.json', 'r'))

BASENAME = config['basename']
VARIANTS = config['variants']
DIST = config["dist"]
OFFSET_X, OFFSET_Y, OFFSET_Z = config["offset"]

ANGLES_Z = config["angles_z"]
ANGLES_X = config["angles_x"]
RENDER_W, RENDER_H = np.array(config["render_dims"]).astype(np.int32)

root_dir = os.getcwd()

for current_dir in VARIANTS:

    print("Switching to directory %s" % current_dir)
    os.chdir(current_dir)
    bpy.ops.wm.open_mainfile(filepath=BASENAME + '.blend')

    bpyscene = bpy.context.scene

    camera = bpy.data.objects['Camera']

    camera.rotation_euler = Euler((90 * np.pi / 180, 0, 0), 'XYZ')
    camera.location = Vector((0, -DIST, 0))

    empty_mesh = bpy.data.meshes.new('Pivot_mesh')
    pivot = bpy.data.objects.new("Pivot", empty_mesh)
    bpyscene.objects.link(pivot)

    camera.parent = pivot

    pivot.location = Vector((OFFSET_X, OFFSET_Y, OFFSET_Z))

    bpy.data.scenes['Scene'].world.horizon_color = Color((0.5, 0.5, 0.5))

    for a_z in ANGLES_Z:
        for a_x in ANGLES_X:

            print("Rendering %s at angles %d %d" % (BASENAME, a_z, a_x))

            pivot.rotation_euler = Euler((a_x * np.pi / 180, 0, a_z * np.pi / 180), 'XYZ')

            output_name = "%s_%d_%d.png" % (BASENAME, a_z, a_x)

            render = bpy.data.scenes['Scene'].render
            render.resolution_x = RENDER_W
            render.resolution_y = RENDER_H
            render.filepath = root_dir + '/renders/' + BASENAME + '/' + current_dir + '/'  + output_name
            bpy.ops.render.render(write_still=True)

    os.chdir(root_dir)
