"""Fixed task sets. Dev seeds are for iterating; eval seeds are the frozen benchmark."""
EVAL_SEEDS = {"easy": range(1000, 1030), "medium": range(1000, 1030), "hard": range(1000, 1030)}
DEV_SEEDS = {"easy": range(0, 200), "medium": range(0, 200), "hard": range(0, 200)}


def task_list(split: str = "eval", difficulties=("easy", "medium", "hard"), limit: int | None = None):
    seeds = EVAL_SEEDS if split == "eval" else DEV_SEEDS
    return [(d, s) for d in difficulties for s in list(seeds[d])[:limit]]


# Campus suite (Berkeley floors, real bag photos): the frozen benchmark is 30 extreme tasks.
CAMPUS_EVAL_SEEDS = {"campus-extreme": range(2000, 2030), "campus-hard": range(2000, 2030)}
CAMPUS_DEV_SEEDS = {"campus-extreme": range(0, 200), "campus-hard": range(0, 200)}


def campus_task_list(split: str = "eval", difficulties=("campus-extreme",), limit: int | None = None):
    seeds = CAMPUS_EVAL_SEEDS if split == "eval" else CAMPUS_DEV_SEEDS
    return [(d, s) for d in difficulties for s in list(seeds[d])[:limit]]
