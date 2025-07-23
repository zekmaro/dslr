TRAIN_DATASET_PATH = "datasets/dataset_train.csv"
TEST_DATASET_PATH = "datasets/dataset_test.csv"
MODEL_DATA_PATH = "shared_data/model.json"
IMAGE_DEST_PATH = "images/pair_plot.png"
MODEL_DATA_PATH = "shared_data/model.json"

LABEL_COLORS = {
	"Gryffindor": "red",
	"Slytherin": "green",
	"Ravenclaw": "blue",
	"Hufflepuff": "orange"
}

LABEL_MAP = {
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
MEAN_CV_THRESHOLD = 6
STD_CV_THRESHOLD = 0.1
CORRELATION_THRESHOLD = 0.9

TRAINING_FEATURES =  [
	# 'Herbology',
	# 'Defense Against the Dark Arts',
	'Divination',
	# 'Muggle Studies',
	'Ancient Runes',
	'History of Magic',
	# 'Transfiguration',
	'Charms',
	# 'Flying',

	# 'Astronomy',
	# 'Arithmancy',
	# 'Care of Magical Creatures',
	# 'Potions'
]
