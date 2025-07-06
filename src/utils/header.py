TRAIN_DATASET_PATH = "datasets/dataset_train.csv"
TEST_DATASET_PATH = "datasets/dataset_test.csv"

HOUSE_COLORS = {
	"Gryffindor": "red",
	"Slytherin": "green",
	"Ravenclaw": "blue",
	"Hufflepuff": "orange"
}

HOUSE_MAP = {
	"Gryffindor": 0,
	"Slytherin": 1,
	"Ravenclaw": 2,
	"Hufflepuff": 3
}

DROP_COLS = [
	"Index", "Hogwarts House", "First Name",
	"Last Name", "Birthday", "Best Hand"
]

COURSES = [
	'Arithmancy', 'Astronomy', 'Herbology',
    'Defense Against the Dark Arts', 'Divination', 
	'Muggle Studies', 'Ancient Runes', 'History of Magic',
	'Transfiguration', 'Potions', 'Care of Magical Creatures',
	'Charms', 'Flying'
]

STAT_INDEXES = [
	'count',
	'mean',
	'std',
	'min',
	'25%',
	'50%',
	'75%', 
	'max'
]

MEAN_THRESHOLD = 1
STD_THRESHOLD = 1
THRESHOLD_MEAN_CV = 6
THRESHOLD_STD_CV = 0.1
