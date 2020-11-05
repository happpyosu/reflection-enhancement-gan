import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def which_gpu_to_use(gpu_index):
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    # avoid all memory to be allocated
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            print("Invalid device or cannot modify virtual devices once initialized.")
            pass

    print("\033[1;34m" + "[gpu]: all GPU devices: " + str(os.environ['CUDA_VISIBLE_DEVICES']) + " \033[0m")
    tf.config.experimental.set_visible_devices(devices=gpus[gpu_index], device_type='GPU')
    print("\033[1;34m" + "[gpu]: config to use " + str(gpu_index) + " gpu only" + " \033[0m")
