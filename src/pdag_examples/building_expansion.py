from typing import Annotated, ClassVar, Literal, get_args, Mapping
import pdag

_TPolicy = Literal[
    "do-nothing", "build-opt-57", "build-opt-33-and-rebuild", "build-exp-33-and-expand"
]
_TAction = Literal[
    "do-nothing", "build-opt-33", "build-opt-57", "tear-down", "expand", "build-exp-33"
]
_TBuildingState = Literal["not-built", "opt-33", "opt-57", "exp-33", "exp-57"]


class BuildingExpansionModel(pdag.Model):
    _POLICIES: ClassVar[tuple[_TPolicy, ...]] = get_args(_TPolicy)
    _ACTIONS: ClassVar[tuple[_TAction, ...]] = get_args(_TAction)
    _BUILDING_STATES: ClassVar[tuple[_TBuildingState, ...]] = get_args(_TBuildingState)

    # Exogenous parameters
    demand = pdag.RealParameter("demand", is_time_series=True, metadata={"XLRM": "X"})
    revenue_per_floor = pdag.RealParameter("revenue_per_floor", metadata={"XLRM": "X"})
    discount_rate = pdag.RealParameter("discount_rate", metadata={"XLRM": "X"})

    # Decision parameters
    policy = pdag.CategoricalParameter(
        "policy",
        _POLICIES,
        metadata={"XLRM": "L"},
    )
    rebuild_threshold = pdag.RealParameter(
        "rebuild_threshold",
        metadata={"XLRM": "L"},
    )
    expand_threshold = pdag.RealParameter(
        "expand_threshold",
        metadata={"XLRM": "L"},
    )

    # Calculated parameters
    action = pdag.CategoricalParameter(
        "action",
        _ACTIONS,
        is_time_series=True,
    )
    building_state = pdag.CategoricalParameter(
        "building_state",
        _BUILDING_STATES,
        is_time_series=True,
    )
    action_cost = pdag.Mapping(
        "action_cost",
        {action: pdag.RealParameter(...) for action in action.categories},
    )
    revenue = pdag.RealParameter("revenue", is_time_series=True)
    cost = pdag.RealParameter("cost", is_time_series=True)
    discount_rate = pdag.RealParameter("discount_rate")
    npv = pdag.RealParameter("npv")

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def determine_action(
        *,
        policy: Annotated[_TPolicy, policy.ref()],
        building_state: Annotated[_TBuildingState, building_state.ref()],
        demand: Annotated[float, demand.ref()],
        rebuild_threshold: Annotated[float, rebuild_threshold.ref()],
        expand_threshold: Annotated[float, expand_threshold.ref()],
    ) -> Annotated[_TAction, action.ref()]: ...

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def state_transition_model(
        *,
        building_state: Annotated[_TBuildingState, building_state.ref()],
        action: Annotated[_TAction, action.ref()],
    ) -> Annotated[_TBuildingState, building_state.ref(next=True)]: ...

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def calculate_revenue(
        *,
        building_state: Annotated[_TBuildingState, building_state.ref()],
        demand: Annotated[float, demand.ref()],
        revenue_per_floor: Annotated[float, revenue_per_floor.ref()],
    ) -> Annotated[float, revenue.ref()]: ...

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def calculate_cost(
        *,
        action: Annotated[_TAction, action.ref()],
        action_cost: Annotated[Mapping[_TAction, float], action_cost.ref()],
    ) -> Annotated[float, cost.ref()]: ...

    @pdag.relationship
    @staticmethod
    def npv_calculation(
        *,
        revenue: Annotated[list[float], revenue.ref(all_time_steps=True)],
        cost: Annotated[list[float], cost.ref(all_time_steps=True)],
        discount_rate: Annotated[float, discount_rate.ref()],
    ) -> Annotated[float, npv.ref()]: ...
