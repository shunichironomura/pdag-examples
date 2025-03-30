from typing import Annotated, ClassVar, Literal, get_args, Mapping
import pdag

TPolicy = Literal[
    "do-nothing", "build-opt-57", "build-opt-33-and-rebuild", "build-exp-33-and-expand"
]
TAction = Literal[
    "do-nothing",
    "build-opt-33",
    "build-opt-57",
    "tear-down-and-build-opt-57",
    "expand",
    "build-exp-33",
]
TBuildingState = Literal["not-built", "opt-33", "opt-57", "exp-33", "exp-57"]


class DemandModel(pdag.Model):
    demand = pdag.RealParameter("demand", is_time_series=True)
    start_time = pdag.RealParameter("start_time")
    ramp_up_time = pdag.RealParameter("ramp_up_time")
    demand_max = pdag.RealParameter("demand_max")

    @pdag.relationship
    @staticmethod
    def demand_model(
        start_time: Annotated[float, start_time.ref()],
        ramp_up_time: Annotated[float, ramp_up_time.ref()],
        demand_max: Annotated[float, demand_max.ref()],
        n_time_steps: Annotated[int, pdag.ExecInfo("n_time_steps")],
    ) -> Annotated[list[float], demand.ref(all_time_steps=True)]: ...


class BuildingExpansionModel(pdag.Model):
    # Enumeration of possible values for policies, actions, and building states
    POLICIES: ClassVar[tuple[TPolicy, ...]] = get_args(TPolicy)
    ACTIONS: ClassVar[tuple[TAction, ...]] = get_args(TAction)
    BUILDING_STATES: ClassVar[tuple[TBuildingState, ...]] = get_args(TBuildingState)

    # Exogenous parameters
    demand_start_time = pdag.RealParameter("demand_start_time")
    demand_ramp_up_time = pdag.RealParameter("demand_ramp_up_time")
    demand_max = pdag.RealParameter("demand_max")
    # Demand is calculated in the DemandModel class
    demand = pdag.RealParameter(
        "demand",
        is_time_series=True,
    )
    revenue_per_floor = pdag.RealParameter("revenue_per_floor")
    discount_rate = pdag.RealParameter("discount_rate")
    action_cost = pdag.Mapping(
        "action_cost",
        {
            "do-nothing": pdag.RealParameter(...),
            "build-opt-33": pdag.RealParameter(...),
            "build-opt-57": pdag.RealParameter(...),
            "tear-down": pdag.RealParameter(...),
            "expand": pdag.RealParameter(...),
            "build-exp-33": pdag.RealParameter(...),
        },
    )

    # Decision parameters
    policy = pdag.CategoricalParameter(
        "policy",
        POLICIES,
    )
    rebuild_threshold = pdag.RealParameter("rebuild_threshold")
    expand_threshold = pdag.RealParameter("expand_threshold")

    # Calculated parameters
    action = pdag.CategoricalParameter(
        "action",
        ACTIONS,
        is_time_series=True,
    )
    building_state = pdag.CategoricalParameter(
        "building_state",
        BUILDING_STATES,
        is_time_series=True,
    )

    revenue = pdag.RealParameter("revenue", is_time_series=True)
    cost = pdag.RealParameter("cost", is_time_series=True)
    npv = pdag.RealParameter("npv")

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def determine_action(
        *,
        policy: Annotated[TPolicy, policy.ref()],
        building_state: Annotated[TBuildingState, building_state.ref()],
        demand: Annotated[float, demand.ref()],
        rebuild_threshold: Annotated[float, rebuild_threshold.ref()],
        expand_threshold: Annotated[float, expand_threshold.ref()],
    ) -> Annotated[TAction, action.ref()]: ...

    @pdag.relationship
    @staticmethod
    def initial_state() -> Annotated[
        TBuildingState, building_state.ref(initial=True)
    ]: ...

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def state_transition_model(
        *,
        building_state: Annotated[TBuildingState, building_state.ref()],
        action: Annotated[TAction, action.ref()],
    ) -> Annotated[TBuildingState, building_state.ref(next=True)]: ...

    demand_model = DemandModel.to_relationship(
        "demand_model",
        inputs={
            DemandModel.start_time.ref(): demand_start_time.ref(),
            DemandModel.ramp_up_time.ref(): demand_ramp_up_time.ref(),
            DemandModel.demand_max.ref(): demand_max.ref(),
        },
        outputs={
            DemandModel.demand.ref(all_time_steps=True): demand.ref(
                all_time_steps=True
            ),
        },
    )

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def calculate_revenue(
        *,
        building_state: Annotated[TBuildingState, building_state.ref()],
        demand: Annotated[float, demand.ref()],
        revenue_per_floor: Annotated[float, revenue_per_floor.ref()],
    ) -> Annotated[float, revenue.ref()]: ...

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def calculate_cost(
        *,
        action: Annotated[TAction, action.ref()],
        action_cost: Annotated[Mapping[TAction, float], action_cost.ref()],
    ) -> Annotated[float, cost.ref()]: ...

    @pdag.relationship
    @staticmethod
    def npv_calculation(
        *,
        revenue: Annotated[list[float], revenue.ref(all_time_steps=True)],
        cost: Annotated[list[float], cost.ref(all_time_steps=True)],
        discount_rate: Annotated[float, discount_rate.ref()],
    ) -> Annotated[float, npv.ref()]: ...
