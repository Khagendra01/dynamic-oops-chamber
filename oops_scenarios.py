"""Deliberately constructed deterministic challenges, separate from holdout seeds."""
from chamber import ChamberEnv, Cuboid


def north_blocked():
    """A central block plus obstruction above (+Y); Z detour remains available."""
    return ChamberEnv(start=(.1,.5,.5),goal=(.9,.5,.5),
        obstacles=(Cuboid('central',(.5,.5,.5),(.18,.20,.24)),
                   Cuboid('north_block',(.25,.69,.5),(.32,.20,.30))),
        radius=.035,dt=.5,max_steps=55)


def south_blocked():
    """Central block plus low-Y and high-Z detour obstructions."""
    return ChamberEnv(start=(.1,.5,.5),goal=(.9,.5,.5),
        obstacles=(Cuboid('central',(.5,.5,.5),(.18,.20,.24)),
                   Cuboid('south_block',(.25,.31,.5),(.32,.20,.30)),
                   Cuboid('top_block',(.25,.5,.71),(.32,.30,.20))),
        radius=.035,dt=.5,max_steps=55)


def moving_gate():
    """A one-axis corridor blocked temporarily by a box crossing +Y."""
    return ChamberEnv(start=(.045,.50,.50),goal=(.85,.50,.50),
        obstacles=(
           Cuboid('lower_wall',(.5,.225,.5),(.96,.45,.96)),
           Cuboid('upper_wall',(.5,.775,.5),(.96,.45,.96)),
           Cuboid('bottom_wall',(.5,.5,.225),(.96,.96,.45)),
           Cuboid('top_wall',(.5,.5,.775),(.96,.96,.45)),
           Cuboid('gate',(.20,.50,.50),(.12,.08,.12),(0,.24,0)),
        ),radius=.035,dt=.5,max_steps=38)
