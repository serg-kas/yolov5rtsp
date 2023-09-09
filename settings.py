"""
Модуль работы с настройками.
Функция get_variable_name достает переменную окружения и возвращает ее значение.
"""
import os
# import json
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
    Если параметр variable - это переменная и есть соответствующее ей значение в переменных окружения,
    то возвращает это значение.
    Если значения нет, то возвращает значение переменной variable (чтобы не "портить"
    значение существующей переменной).
    Если variable - имя переменной (строка, а не переменная) и есть соответствующее ей значение
    в переменных окружения, то возвращает это значение. Если значения нет, то возвращает default_value

    :param variable: существующая переменная или имя переменной (строка)
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
            value = [item.strip() for item in value.split(',')]
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
            print("  Используем значение {} по умолчанию: {}".format(variable, default_value))
            return default_value


# ################## ОБЩИЕ ПАРАМЕТРЫ ######################
# Флаг отладки
DEBUG = get_value_from_env("DEBUG", default_value=False)




# Показывать видео на экране
SHOW_VIDEO = get_value_from_env("SHOW_VIDEO", default_value=False)

# Транслировать видео на rtsp сервер
VIDEO_TO_RTSP = get_value_from_env("VIDEO_TO_RTSP", default_value=False)

# Целевая ширина фрейма для обработки и показа изображения
def_W = get_value_from_env("DEF_W", default_value=800)

# Телеграм
bot_Token = get_value_from_env("BOT_TOKEN")
chat_Id = get_value_from_env("CHAT_ID")
url_tg = get_value_from_env("URL_TG", default_value=f"https://api.telegram.org/bot{bot_Token}/sendMessage?chat_id={chat_Id}&text=")
if bot_Token is None or chat_Id is None:
    print("bot_Token и chat_Id - обязательные параметры")
    exit(1)

# RTSP для получения и трансляции видео
RTSP_URL = get_value_from_env("RTSP_URL")
RTSP_SERVER = get_value_from_env("RTSP_SERVER")
if VIDEO_TO_RTSP and RTSP_SERVER is None:
    print("RTSP_server - необходимый параметр для трансляции")
    exit(1)

# Попыток соединиться с источником видео
ATTEMPTS = get_value_from_env("ATTEMPTS", default_value=3)

# ############### ПАРАМЕТРЫ работы с НС ###################
# Имена классов, которые детектим.
NAMES_TO_DETECT = get_value_from_env("NAMES_TO_DETECT", default_value=['person', 'bottle', 'cell phone', 'chair'])

# Паттерн (сочетание классов), который ищем
PATTERN_NAMES = get_value_from_env("PATTERN_NAMES", default_value=['person', 'chair'])

# Confidence и IOU модели
MODEL_CONF = get_value_from_env("MODEL_CONF", default_value=0.25)
MODEL_IOU = get_value_from_env("MODEL_IOU", default_value=0.45)

# Параметры обобщения (аккумулирования) предиктов
N_ACC = get_value_from_env("N_ACC", default_value=10)     # размер аккумулятора, предиктов
N_PRED = get_value_from_env("N_PRED", default_value=2)    # при скольких предиктах в аккумуляторе показываем объект
N_COORD = get_value_from_env("N_COORD", default_value=3)  # по скольки последним предиктам усредняем bb (координаты)

# Параметры IOU трекера
IOU_to_track = get_value_from_env("IOU_TO_TRACK", default_value=0.4)    # порог IOU сохранить трек объекта
IOU_to_rename = get_value_from_env("IOU_TO_RENAME", default_value=0.9)  # порог IOU переименовать объект
