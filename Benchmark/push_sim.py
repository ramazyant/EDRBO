"""Headless Python-3 port of the 3-D robot-pushing simulator of
Wang & Jegelka (2017, "Max-value Entropy Search") / Wang et al. (2018, "Ensemble BO").
Original code: https://github.com/zi-w/Max-value-Entropy-Search (MIT License, (c) 2017 Zi Wang),
files test_functions/python_related/push_world.py and generate_simudata3.py.

Changes w.r.t. the original: no pygame/GUI, Python 3, the velocity noise is an argument
(push_noise=0.0 gives a deterministic simulator; the original used 0.1) and arctan2 is used
instead of arctan(ry/rx) only to avoid the division by zero at rx = 0 (the hand angle is
otherwise identical modulo pi, which leaves the rectangular hand unchanged).
"""
import numpy as np
from Box2D import b2World, b2PolygonShape, b2CircleShape, b2Vec2

_TARGET_FPS = 100
_TIME_STEP = 1.0 / _TARGET_FPS
_VEL_ITERS, _POS_ITERS = 10, 10


def push_object_final_position(rx, ry, duration, push_noise=0.0, rng=None):
    """Simulate one push and return the object's final (x, y) position.

    rx, ry   : initial hand position in [-5, 5]^2 (object starts at the origin)
    duration : push duration in [1, 30]; the simulator runs int(10 * duration) steps
    """
    rng = np.random.default_rng() if rng is None else rng
    world = b2World(gravity=(0.0, 0.0), doSleep=True)
    # table + object (circle, radius 1, friction .01, density .05) -- as in generate_simudata3.py
    base = world.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(box=(500, 500)))
    thing = world.CreateDynamicBody(position=(0, 0))
    thing.CreateFixture(shape=b2CircleShape(radius=1), density=0.05, friction=0.01)
    world.CreateFrictionJoint(bodyA=base, bodyB=thing, maxForce=5, maxTorque=2)
    # hand (rectangle 0.3 x 1, density .1, friction .1)
    if abs(rx) < 1e-12 and abs(ry) < 1e-12:
        rx = 1e-6
    hand = world.CreateDynamicBody(position=(rx, ry), angle=float(np.arctan2(ry, rx)))
    hand.CreateFixture(shape=b2PolygonShape(box=(0.3, 1)), density=0.1, friction=0.1)
    world.CreateFrictionJoint(bodyA=base, bodyB=hand, maxForce=2, maxTorque=2)
    forceunit = 30.0

    steps = int(float(duration) * 10)
    desired = np.array(thing.position) - np.array(hand.position)
    desired = desired / np.linalg.norm(desired) * 5.0
    if push_noise > 0:
        desired = desired + rng.normal(0.0, push_noise, size=2)
    rvel = b2Vec2(float(desired[0]), float(desired[1]))
    for t in range(steps + 100):
        if t < steps:
            # apply_wrench(rlvel=rvel, ravel=0) of the original end_effector
            avel = hand.angularVelocity
            hand.ApplyTorque(hand.mass * (0.0 - avel) * 30.0, wake=True)
            lvel = hand.linearVelocity
            force = hand.mass * (rvel - b2Vec2(lvel)) * forceunit
            hand.ApplyForce(force, hand.position, wake=True)
        world.Step(_TIME_STEP, _VEL_ITERS, _POS_ITERS)
    return np.array([thing.position[0], thing.position[1]], dtype=float)
