import numpy as np

def generate_snake_mjcf(n_segments, total_length=1.0):
    """
    Generate MJCF XML string for a planar snake robot.
    
    Args:
        n_segments (int): Number of body segments (excluding head)
        total_length (float): Total length of the snake robot
    """
    # Calculate segment length to maintain fixed total length
    segment_length = total_length / (n_segments + 1)  # +1 for head
    
    mjcf = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="planar_snake_robot">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="1" density="5.0" friction="1.5 0.1 0.1" margin="0.01"/>
    </default>

    <option timestep="0.01" integrator="RK4">
        <flag warmstart="enable"/>
    </option>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" pos="0 0 0" size="50 50 0.1" conaffinity="1" rgba="0.8 0.9 0.8 1" condim="1"/>
        
        <!-- Planar Snake body -->
        <body name="head" pos="0 0 {segment_length/2}">
            <freejoint name="root"/>
            <geom name="head" type="capsule" size="{segment_length/4} {segment_length/2}" rgba="0.7 0.7 0 1"/>
'''
    
    # Generate body segments
    current_segment = '''            <body name="segment{}" pos="0 0 -{}">\n'''
    segment_template = '''                <joint name="joint{}" type="hinge" axis="0 0 1" limited="true" range="-30 30"/>
                <geom name="segment{}" type="capsule" size="{} {}" rgba="0.7 0.7 0 1"/>\n'''
    
    # Add segments recursively
    indent = "            "
    for i in range(n_segments):
        mjcf += current_segment.format(i+1, segment_length)
        mjcf += indent + segment_template.format(
            i+1, i+1,
            segment_length/4,  # radius
            segment_length/2   # half-length
        )
        indent += "    "
    
    # Close all body tags
    mjcf += indent + "</body>\n" * n_segments
    
    # Add actuators
    mjcf += '''        </body>
    </worldbody>

    <actuator>\n'''
    
    # Add motors for each joint
    for i in range(1, n_segments + 1):
        mjcf += f'        <motor joint="joint{i}" gear="100" name="motor{i}"/>\n'
    
    # Close remaining tags
    mjcf += '''    </actuator>
</mujoco>'''

    return mjcf