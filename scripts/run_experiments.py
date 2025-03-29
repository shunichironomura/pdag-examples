from itertools import product

import capsula
import numpy as np
import pdag
from pdag._experiment.results import generate_random_string
from rich import print  # noqa: A004

from pdag_examples.building_expansion import BuildingExpansionModel
from pdag_examples.utils import distance_constrained_sampling


@capsula.run()
@capsula.context(capsula.FileContext.builder("results.parquet", move=True), mode="post")
@capsula.context(capsula.FunctionContext.builder(), mode="pre")
def run_experiments(
    *,
    n_scenarios: int,
    n_decisions: int,
    n_time_steps: int,
    np_rng_seed: int,
    uniform_in_distance: bool,
) -> None:
    core_model = BuildingExpansionModel.to_core_model()
    exec_model = pdag.create_exec_model_from_core_model(
        core_model=core_model, n_time_steps=n_time_steps
    )

    input_parameters = exec_model.input_parameters()
    scenario_parameters = {
        parameter_id: parameter
        for parameter_id, parameter in input_parameters.items()
        if parameter.metadata.get("XLRM", None) == "X"
    }
    decision_parameters = {
        parameter_id: parameter
        for parameter_id, parameter in input_parameters.items()
        if parameter.metadata.get("XLRM", None) == "L"
    }
    print("Scenario parameters:")
    print(scenario_parameters)
    print("Decision parameters:")
    print(decision_parameters)
    assert set(scenario_parameters) | set(decision_parameters) == set(input_parameters)

    rng = np.random.default_rng(seed=np_rng_seed)
    scenarios = distance_constrained_sampling(
        scenario_parameters,
        n_samples=n_scenarios,
        budget=1,
        rng=rng,
        uniform_in_distance=uniform_in_distance,
    )
    decisions = pdag.sample_parameter_values(
        decision_parameters, n_samples=n_decisions, rng=rng
    )

    cases = (
        scenario | decision for scenario, decision in product(scenarios, decisions)
    )
    scenario_ids = [generate_random_string() for _ in range(n_scenarios)]
    decision_ids = [generate_random_string() for _ in range(n_decisions)]
    metadata = (
        {"scenario_id": scenario_id, "decision_id": decision_id}
        for scenario_id, decision_id in product(scenario_ids, decision_ids)
    )

    results = pdag.run_experiments(
        exec_model, cases, metadata=metadata, n_cases=n_scenarios * n_decisions
    )
    print(results)
    results.write_parquet("results.parquet")


if __name__ == "__main__":
    run_experiments(
        n_scenarios=1_000,
        n_decisions=1_000,
        np_rng_seed=42,
        uniform_in_distance=True,
    )
