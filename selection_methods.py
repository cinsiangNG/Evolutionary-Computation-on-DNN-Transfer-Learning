import random

from .individual import GaIndividual


class SelectionMethods:
    @classmethod
    def tournament_selection(
        cls, population: list[GaIndividual], tournament_size: int
    ) -> GaIndividual:
        tournament_pool = random.sample(population, tournament_size)
        return max(tournament_pool, key=lambda ind: ind.fitness)

    @classmethod
    def roulette_wheel_selection(cls, population: list[GaIndividual]) -> GaIndividual:
        weights = [ind.fitness for ind in population]
        selected = random.choices(population, weights=weights, k=1)
        return selected[0]

    @classmethod
    def rank_selection(cls, population: list[GaIndividual]) -> GaIndividual:
        population.sort(key=lambda ind: ind.fitness)
        rank_probabilities = [
            i / len(population) for i in range(1, len(population) + 1)
        ]
        selected = random.choices(population, weights=rank_probabilities, k=1)
        return selected[0]

    @classmethod
    def stochastic_universal_sampling(
        cls, population: list[GaIndividual], offspr_sz: int
    ) -> GaIndividual:
        total_fitness = sum(ind.fitness for ind in population)
        interval = total_fitness / offspr_sz
        start_point = random.uniform(0, interval)
        selected = population[0]
        current_point = start_point

        for ind in population:
            current_point += ind.fitness
            if current_point > total_fitness:
                selected = ind
                break

        return selected

    @classmethod
    def random_selection(cls, population: list[GaIndividual]) -> GaIndividual:
        return random.choice(population)

    @classmethod
    def linear_ranking_selection(cls, population: list[GaIndividual]) -> GaIndividual:
        population.sort(
            key=lambda ind: ind.fitness
        )  # Use list.sort() for in-place sorting
        rank_probabilities = [
            2 * (i + 1) / (len(population) * (len(population) + 1))
            for i in range(len(population))
        ]
        selected = random.choices(population, weights=rank_probabilities, k=1)
        return selected[0]

    @classmethod
    def best_of_tournament_selection(
        cls,
        population: list[GaIndividual],
        num_tournaments: int = 3,
        tournament_size: int = 2,
    ) -> GaIndividual:
        best_individual = population[0]
        for _ in range(num_tournaments):
            tournament_pool = random.sample(population, tournament_size)
            tournament_winner = max(tournament_pool, key=lambda ind: ind.fitness)

            if tournament_winner.fitness > best_individual.fitness:
                best_individual = tournament_winner

        return best_individual

    @classmethod
    def self_adaptive_selection(
        cls, population: list[GaIndividual], selection_pressure: float = 0.7
    ) -> GaIndividual:
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        selected = random.sample(
            population[: int(selection_pressure * len(population))], k=1
        )
        return selected[0]

    @classmethod
    def power_selection(
        cls, population: list[GaIndividual], power: float = 2
    ) -> GaIndividual:
        probabilities = [(ind.fitness) ** power for ind in population]
        selected = random.choices(population, weights=probabilities, k=1)
        return selected[0]

    @classmethod
    def threshold_selection(
        cls, population: list[GaIndividual], threshold: float = 0.8
    ) -> GaIndividual:
        eligible_individuals = [ind for ind in population if ind.fitness >= threshold]
        selected = random.sample(eligible_individuals, k=1)
        return selected[0]

    @classmethod
    def randomized_top_k_selection(
        cls, population: list[GaIndividual], top_k: int = 5
    ) -> GaIndividual:
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        top_performers = population[:top_k]
        selected = random.choice(top_performers)
        return selected

    @classmethod
    def age_based_selection(
        cls, population: list[GaIndividual], age_weight=0.1
    ) -> GaIndividual:
        ages = [ind.age for ind in population]
        fitness_scores = [ind.fitness for ind in population]
        combined_scores = [
            fitness + age_weight * age for fitness, age in zip(fitness_scores, ages)
        ]
        selected = random.choices(population, weights=combined_scores, k=1)
        return selected[0]

    @classmethod
    def soft_ranking_selection(cls, population: list[GaIndividual]) -> GaIndividual:
        population.sort(key=lambda ind: ind.fitness)
        rank_probabilities = [
            i / len(population) for i in range(1, len(population) + 1)
        ]
        weights = [rp * ind.fitness for rp, ind in zip(rank_probabilities, population)]
        selected = random.choices(population, weights=weights, k=1)
        return selected[0]
