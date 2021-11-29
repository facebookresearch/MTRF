# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

from pathlib import Path
from gym.envs.registration import register
# export PYTHONPATH="/Users/vikashplus/Libraries/MTRF/MTRF/r3l/::PYTHONPATH"

from r3l.r3l_envs.inhand_env.base import ObjectType

PROJECT_PATH = str(Path(__file__).parent.parent)
RESET_STATES_PATH = str(Path(PROJECT_PATH) / "r3l/r3l_agents/reset_states")

# Pincer envs --------------------------------
HORIZON = 100

def register_with_eval_env(id="", suffix="Eval", **kwargs):
    assert "-" in id, "Environment ID must be of the form <domain><task>-<version>"
    register(id=id, **kwargs)
    eval_id = id.replace("-", suffix + "-")
    eval_kwargs = kwargs.copy()
    if "kwargs" not in eval_kwargs:
        eval_kwargs["kwargs"] = {}
    eval_kwargs["kwargs"].update(reset_every_n_episodes=1)
    register(id=eval_id, **eval_kwargs)

# Rotate
register(
    id='SawyerDhandInHandRotateFixed-v0',
    entry_point='r3l.r3l_envs.inhand_env.rotate:SawyerDhandInHandObjectReorientMidairFixed',
    max_episode_steps=HORIZON,
)

# Flip Up
register(
    id='SawyerDhandInHandFlipUpFixed-v0',
    entry_point='r3l.r3l_envs.inhand_env.flipup:SawyerDhandInHandObjectFlipUpFixed',
    # entry_point='r3l.r3l_envs.inhand_env.flipup:SawyerDhandInHandFlipUpEnvFixed',
    max_episode_steps=HORIZON,
)

# Reach
register(
    id='SawyerDhandInHandReachFixed-v0',
    entry_point='r3l.r3l_envs.inhand_env.reach:SawyerDhandInHandReachFixed',
    max_episode_steps=HORIZON,
)
register(
    id='SawyerDhandInHandReachResetFree-v0',
    entry_point='r3l.r3l_envs.inhand_env.reach:SawyerDhandInHandReachResetFree',
    max_episode_steps=HORIZON,
)
register(
    id='SawyerDhandInHandReachRandom-v0',
    entry_point='r3l.r3l_envs.inhand_env.reach:SawyerDhandInHandReachEnvRandom',
    max_episode_steps=HORIZON,
)

# Combined envs ================================================================
register(
    id='SawyerDhandInHandManipulateResetFree-v0',
    entry_point='r3l.r3l_envs.inhand_env.manipulate_resetfree:SawyerDhandInHandManipulateResetFree',
    max_episode_steps=HORIZON,
)


from r3l.r3l_envs.inhand_env.base import OBJECT_TYPE_TO_MODEL_PATH
for object_type in OBJECT_TYPE_TO_MODEL_PATH:
    object_name = object_type.name
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}PoseFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.pose:SawyerDhandInHandObjectPoseFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.reposition:SawyerDhandInHandObjectRepositionFixed',
        kwargs={
            "object_type": object_type
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionRandomInit-v0',
        entry_point='r3l.r3l_envs.inhand_env.reposition:SawyerDhandInHandObjectRepositionRandomInit',
        kwargs={
            "object_type": object_type
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionCornerToCorner-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectRepositionCornerToCorner',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionCenterToRandom-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectRepositionCenterToRandom',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientCenterToRandom-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectReorientCenterToRandom',
        kwargs={
            "object_type": object_type,
        }
    )

    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.reorient:SawyerDhandInHandObjectReorientFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    # Same as above, without any reposition reward
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientOnlyFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.reorient:SawyerDhandInHandObjectReorientFixed',
        kwargs={
            "object_type": object_type,
            "reorient_only": True,
        }
    )
    register_with_eval_env(
        id='SawyerDhandInHand{}PickupFixed-v0'.format(object_name),
        entry_point='r3l.r3l_envs.inhand_env.pickup:SawyerDhandInHandObjectPickupFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}FlipDownFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.flipdown:SawyerDhandInHandObjectFlipDownFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}FlipUpFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.flipup:SawyerDhandInHandObjectFlipUpFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}FlipUpResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.flipup:SawyerDhandInHandObjectFlipUpResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}FlipUpResetFreeBaseline-v0',
        entry_point='r3l.r3l_envs.inhand_env.flipup:SawyerDhandInHandObjectFlipUpResetFreeBaseline',
        kwargs={
            "object_type": object_type,
        }
    )

    # =============== Phased Environments =============== #
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionReorientPerturbResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectRepositionReorientPerturbResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionReorientPickupPerturbResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectRepositionReorientPickupPerturbResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}AllPhasesResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectAllPhasesResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}AllPhasesResetFree-v1',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase:SawyerDhandInHandObjectAllPhasesResetFree_v1',
        kwargs={
            "object_type": object_type,
        }
    )

    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientOnlyMidairFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.reorient_midair:SawyerDhandInHandObjectReorientOnlyMidairFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientMidairPhased-v0',
        entry_point='r3l.r3l_envs.inhand_env.reorient_midair:SawyerDhandInHandObjectReorientMidairPhased',
        kwargs={
            "object_type": object_type,
        }
    )

    # Baselines
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}AllPhasesResetFreeNoPerturb-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase_baselines:SawyerDhandInHandObjectAllPhasesResetFreeNoPerturb',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionReorientResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase_baselines:SawyerDhandInHandObjectRepositionReorientResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionReorientPickupResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase_baselines:SawyerDhandInHandObjectRepositionReorientPickupResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionPickupResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase_baselines:SawyerDhandInHandObjectRepositionPickupResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}PickupFlipUpFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase_baselines:SawyerDhandInHandObjectPickupFlipUpFixed',
        kwargs={
            "object_type": object_type,
        }
    )

    # ============== Midair Tasks ============== #
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientMidairFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.rotate:SawyerDhandInHandObjectReorientMidairFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionMidairFixed-v0',
        entry_point='r3l.r3l_envs.inhand_env.rotate:SawyerDhandInHandObjectRepositionMidairFixed',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientMidairResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.rotate:SawyerDhandInHandObjectReorientMidairResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}PickupFlipUpReorientMidairResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.multi_phase_baselines:SawyerDhandInHandObjectPickupFlipUpReorientMidairResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}ReorientMidairSlottedResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.rotate:SawyerDhandInHandObjectReorientMidairSlottedResetFree',
        kwargs={
            "object_type": object_type,
        }
    )
    register_with_eval_env(
        id=f'SawyerDhandInHand{object_name}RepositionMidairSlottedResetFree-v0',
        entry_point='r3l.r3l_envs.inhand_env.rotate:SawyerDhandInHandObjectRepositionMidairSlottedResetFree',
        kwargs={
            "object_type": object_type,
        }
    )

register_with_eval_env(
    id="SawyerDhandInHandDodecahedronPalmDownRepositionMidairFixed-v0",
    entry_point='r3l.r3l_envs.inhand_env.basket:SawyerDhandInHandDodecahedronBasketFixed',
    kwargs={
        'object_type': ObjectType.Dodecahedron,
    }
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBasketFixed-v0",
    entry_point='r3l.r3l_envs.inhand_env.basket:SawyerDhandInHandDodecahedronBasketFixed',
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBasketResetFree-v0",
    entry_point='r3l.r3l_envs.inhand_env.basket:SawyerDhandInHandDodecahedronBasketResetFree',
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBasketDropFixed-v0",
    entry_point='r3l.r3l_envs.inhand_env.basket:SawyerDhandInHandDodecahedronBasketDropFixed',
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBasketResetController-v0",
    entry_point='r3l.r3l_envs.inhand_env.basket:SawyerDhandInHandDodecahedronBasketResetController',
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBasketPhased-v0",
    entry_point='r3l.r3l_envs.inhand_env.basket:SawyerDhandInHandDodecahedronBasketPhased',
    max_episode_steps=HORIZON,
)

register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBulbFixed-v0",
    entry_point='r3l.r3l_envs.inhand_env.bulb:SawyerDhandInHandDodecahedronBulbFixed',
    max_episode_steps=100,
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBulbResetFree-v0",
    entry_point='r3l.r3l_envs.inhand_env.bulb:SawyerDhandInHandDodecahedronBulbResetFree',
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBulbResetController-v0",
    entry_point='r3l.r3l_envs.inhand_env.bulb:SawyerDhandInHandDodecahedronBulbResetController',
)
register_with_eval_env(
    id="SawyerDhandInHandDodecahedronBulbPhased-v0",
    entry_point='r3l.r3l_envs.inhand_env.bulb:SawyerDhandInHandDodecahedronBulbPhased',
    max_episode_steps=HORIZON,

)