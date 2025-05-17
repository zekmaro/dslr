from typing import Any


def calculate_mean(*args: Any) -> float:
	"""Calculate the mean of a list of numbers."""
	if len(args) == 0:
		return 0
	return sum(args) / len(args)


def calculate_median(*args: Any) -> float:
	"""Calculate the median of a list of numbers."""
	sorted_args = sorted(args)
	n = len(sorted_args)
	if n % 2 == 0:
		return (sorted_args[n // 2 - 1] + sorted_args[n // 2]) / 2
	else:
		return sorted_args[n // 2]


def calculate_quartile(*args: Any) -> tuple[float, float]:
	"""Calculate the quartiles of a list of numbers."""
	sorted_args = sorted(args)
	n = len(sorted_args)
	q1 = sorted_args[n // 4]
	q3 = sorted_args[3 * n // 4]
	return [q1, q3]


def calculate_variance(*args: Any) -> float:
	"""Calculate the variance of a list of numbers."""
	if len(args) == 0:
		return 0
	mean = calculate_mean(*args)
	return sum((x - mean) ** 2 for x in args) / len(args)


def calculate_stddev(*args: Any) -> float:
	"""Calculate the standard deviation of a list of numbers."""
	if len(args) == 0:
		return 0
	variance = calculate_variance(*args)
	return variance ** 0.5


def ft_statistics(*args: Any, **kwargs: Any) -> None:
	"""Calculate and print various statistics based on the provided arguments."""
	if len(args) == 0:
		print("ERROR")
		return

	measures_methods = {
		"mean": calculate_mean(*args),
		"median": calculate_median(*args),
		"quartile": calculate_quartile(*args),
		"std": calculate_stddev(*args),
		"var": calculate_variance(*args)
	}

	for value in kwargs.values():
		if value in measures_methods:
			print(f"{value} : {measures_methods[value]}")
		else:
			print("ERROR")
