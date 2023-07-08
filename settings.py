"""
Модуль работы с настройками.
Функция get_variable_name достает из переменную окружения и возвращает ее значение.
"""
import os
import json
from dotenv import load_dotenv

# При наличии файла .env загружаем из него переменные окружения .env
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    print("Загружаем переменные окружения из файла .env")
    load_dotenv(dotenv_path)


def get_variable_name(variable):
    """
    Возвращает имя переменной как строку
    :param variable: переменная
    :return: имя переменной
    """
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def get_value_from_env(variable, default_value=None, prefix_name="APP_", verbose=False):
    """
    Ищет значение в переменных окружения.
    Если параметр variable - переменная и есть соответствующее значение в переменных окружения,
    то возвращает это значение. Если значения нет, то возвращает значение переменной variable.
    Если variable - имя переменной (строка, а не переменная) и есть соответствующее значение
    в переменных окружения, то возвращает это значение. Если значения нет, то возвращает default_value

    :param variable: существующая переменная или строка (имя переменной)
    :param default_value: значение по умолчанию
    :param prefix_name: префикс прибавляется к имени переменной
    :param verbose: выводить подробные сообщения
    :return: значение переменной
    """
    variable_name = get_variable_name(variable)
    if variable_name != 'variable':
        got_variable = True
    else:
        got_variable = False
        variable_name = variable
    #
    value = os.getenv(prefix_name + variable_name.upper())
    if value is not None:
        if type(default_value) is bool:
            value = bool(value)
        elif type(default_value) is int:
            value = int(value)
        elif type(default_value) is float:
            value = float(value)
        elif type(default_value) is list:
            value = json.loads(value)
        #
        print("  Получили значение из переменной окружения: {}={}".format(variable_name, value))
        # print(variable_name, value, type(value))
        return value
    else:
        if verbose:
            print("  Не найдено значения переменной {} в переменных окружения".format(variable_name))
        if got_variable:
            if verbose:
                print("  Оставляем значение переменной без изменения: {}".format(variable))
            return variable
        else:
            if verbose:
                print("  Используем значение по умолчанию: {}".format(default_value))
            return default_value


# ################## ОБЩИЕ ПАРАМЕТРЫ ######################
# Флаг отладки
DEBUG = get_value_from_env("DEBUG", default_value=False)

# Показывать видео на экране
SHOW_VIDEO = get_value_from_env("SHOW_VIDEO", default_value=False)

# Транслировать видео на rtsp сервер
VIDEO_to_RTSP = get_value_from_env("VIDEO_TO_RTSP", default_value=False)

# Целевая ширина фрейма для обработки и показа изображения
def_W = get_value_from_env("def_w", default_value=800)

# Телеграм
bot_Token = os.getenv('bot_token')
chat_Id = os.getenv('chat_id')
url_tg = os.getenv('URL_TG')
if (bot_Token or chat_Id) is None:
    print("bot_Token и chat_Id - обязательные параметры")
    exit(1)

# RTSP для получения и трансляции видео
RTSP_URL = os.getenv('RTSP_URL')
RTSP_server = os.getenv('rtsp_server')
if VIDEO_to_RTSP and RTSP_server is None:
    print("RTSP_server - необходимый параметр для трансляции")
    exit(1)

# ############### ПАРАМЕТРЫ работы с НС ###################
# Имена классов, которые детектим.
NAMES_TO_DETECT = get_value_from_env("NAMES_TO_DETECT", default_value=['person', 'bottle', 'cell phone', 'chair'])

# Паттерн (сочетание классов), который ищем
PATTERN_NAMES = get_value_from_env("PATTERN_NAMES", default_value=['person', 'cell phone'])



# Флаг вывода подробных сообщений в консоль
VERBOSE = get_value_from_env("VERBOSE", default_value=False)

# Папки по умолчанию
SOURCE_PATH = 'source_files'
OUT_PATH = 'out_files'
MODELS_PATH = 'models'
EMB_PATH = get_value_from_env("EMB_PATH", default_value='data')
TEMPL_PATH = get_value_from_env("TEMPLATES", default_value='templates')

# Допустимые форматы изображений для загрузки в программу
ALLOWED_IMAGES = ['.jpg', '.jpeg', '.png']
# Допустимые форматы файлов для загрузки в программу
ALLOWED_TYPES = ALLOWED_IMAGES + ['.pdf']

# Ширина консоли, символов
CONS_COLUMNS = 0  # 0 = попытаться определить автоматически

# ################## ПАРАМЕТРЫ МОДЕЛЕЙ ####################
# Модель OD Stamp для вызова из opencv
MODEL_DNN_FILE = 'models/best.onnx'
FORCE_CUDA = False
INPUT_HEIGHT = 416
INPUT_WIDTH = 416
CONFIDENCE_THRESHOLD = 0.35  # для одиночных печатей можно 0.8 или 0.9
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.45

# Модель CNN классификатор isText для вызова из opencv
MODEL_DNN_FILE_txt = 'models/cnn_txt.onnx'
FORCE_CUDA_txt = False
INPUT_HEIGHT_txt = 55
INPUT_WIDTH_txt = 1030
CONFIDENCE_THRESHOLD_txt = 0.80  # 0.70 -> 0.75 -> 0.80

# Модель CNN экстрактора фич для вызова из opencv
MODEL_DNN_FILE_fe = 'models/vgg16fe.onnx'
FORCE_CUDA_fe = False
INPUT_HEIGHT_fe = 224
INPUT_WIDTH_fe = 224
# и ее ансамблевая обработка
N_votes_fe = 10
CONFIDENCE_THRESHOLD_votes = 0.30
# Собирать результаты предиктов классификатора в новый датасет
CNNFE_DATA_COLLECT = True
CNNFE_DATA_COLLECT_path = 'data_collect'
CNNFE_DATA_COLLECT_limit = 200  # 0 = не ограничивать количество
# Параметры режима пересчета эмбеддингов (rebuild_emb)
FORCE_PREPROCESS_IMG = get_value_from_env("FORCE_PREPROCESS_IMG", default_value=False)
PREP_SCATTER_ONLY = get_value_from_env("PREP_SCATTER_ONLY", default_value=False)
FOLDERS_TO_PROCESS = get_value_from_env("FOLDERS_TO_PROCESS", default_value=[])

# Модель CNN_mnist классификатора mnist
MODEL_DNN_FILE_mnist = 'models/cnn_mnist.onnx'
FORCE_CUDA_mnist = False
INPUT_HEIGHT_mnist = 28
INPUT_WIDTH_mnist = 28

# Модель (reader) EasyOCR
FORCE_CUDA_easyocr = get_value_from_env("FORCE_CUDA_EASYOCR", default_value=True)

# ################### ПАРАМЕТРЫ pdf2img ###################
DPI = 150
# Разрешение DPI==150 дает размеры
SCAN_MIN_SIZE = 1241
SCAN_MAX_SIZE = 1755


# ####################### Цвета RGB #######################
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
purple = (255, 0, 255)
turquoise = (255, 255, 0)
white = (255, 255, 255)
