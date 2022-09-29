import random
from random import randrange
from time import time
import numpy as np
import pandas as pd
import glob
import enum
import os
import matplotlib.pyplot as plt


class Basic_Stat(enum.Enum):
    velocity = 0
    acceleration = 1
    angular_velocity = 4


def create_ang_vel_histograms(folder_path: str) -> None:
    """Creates angular velocity histograms for each driver (folder)

    :param folder_path: str - path to folder containing all drivers' trip data
    """
    histogram_size = 400
    folder_list = glob.glob(folder_path + '/*')

    print('Creating Angular Velocity Histograms')
    for i in range(len(folder_list)):
        print(i)
        file_list = glob.glob(folder_list[i] + '/*.csv')
        ang_vel_hist = [0] * histogram_size

        for file in file_list:
            ang_vel_series = pd.read_csv(file).to_numpy()[
                Basic_Stat.angular_velocity.value]
            ang_vel_series = ang_vel_series.astype(float)
            for k in range(len(ang_vel_series)):
                index = ((ang_vel_series[k] + 180) % 360)
                if (index > 178 and index <= 182):
                    index = round((index * 100) + 200) % 400
                    ang_vel_hist[index] += 1

        sum = 0
        for j in range(histogram_size):
            sum += ang_vel_hist[j]
        ang_vel_hist = np.divide(ang_vel_hist, sum)
        ang_vel_hist = np.multiply(ang_vel_hist, 100)

        np.savetxt('data/angular-vel-hist/driver' + str(i) + '.csv',
                   ang_vel_hist, fmt='%5f', delimiter=',')


def create_accel_histograms(folder_path: str) -> None:
    """Creates acceleration histograms for each driver (folder)

    :param folder_path: str - path to folder containing all drivers' trip data
    """
    histogram_size = 100
    folder_list = glob.glob(folder_path + '/*')

    print('Creating Acceleration Histograms')
    for i in range(len(folder_list)):
        print(i)
        file_list = glob.glob(folder_list[i] + '/*.csv')
        accel_hist = [0] * histogram_size

        for file in file_list:
            accel_series = pd.read_csv(file).to_numpy()[
                Basic_Stat.acceleration.value]
            accel_series = np.multiply(accel_series, 10)
            accel_series = np.floor(accel_series).astype(int)
            for k in range(len(accel_series)):
                index = (accel_series[k] + 50)
                if (index >= 0 and index < 100):
                    accel_hist[index] += 1

        sum = 0
        for j in range(histogram_size):
            sum += accel_hist[j]
        accel_hist = np.divide(accel_hist, sum)
        accel_hist = np.multiply(accel_hist, 100)

        np.savetxt('data/accel-hist/driver' + str(i) + '.csv',
                   accel_hist, fmt='%5f', delimiter=',')


def create_combined_accel_histograms(folder_path: str) -> None:
    """Creates an acceleration histogram with combined positive and negative values

    :param folder_path : str - path to folder containing all drivers' trip data
    """
    histogram_size = 50
    folder_list = glob.glob(folder_path + '/*')

    print('Creating Acceleration Histograms')
    for i in range(len(folder_list)):
        print(i)
        file_list = glob.glob(folder_list[i] + '/*.csv')
        accel_hist = [0] * histogram_size

        for file in file_list:
            accel_series = pd.read_csv(file).to_numpy()[
                Basic_Stat.acceleration.value]
            accel_series = np.multiply(accel_series, 10)
            accel_series = np.floor(accel_series).astype(int)
            accel_series = np.absolute(accel_series)
            for k in range(len(accel_series)):
                index = (accel_series[k])
                if (index >= 0 and index < 50):
                    accel_hist[index] += 1

        sum = 0
        for j in range(histogram_size):
            sum += accel_hist[j]
        accel_hist = np.divide(accel_hist, sum)
        accel_hist = np.multiply(accel_hist, 100)

        np.savetxt('data/combined-accel-hist/driver' + str(i) + '.csv',
                   accel_hist, fmt='%5f', delimiter=',')


def create_vel_histograms(folder_path: str) -> None:
    """Creates velocity histograms for each driver (folder)

    :param folder_path: str - path to folder containing all drivers' trip data
    """
    histogram_size = 400
    folder_list = glob.glob(folder_path + '/*')

    print('Creating Velocity Histograms')
    for i in range(len(folder_list)):
        print(i)
        file_list = glob.glob(folder_list[i] + '/*.csv')
        vel_hist = [0] * histogram_size

        for file in file_list:
            vel_series = pd.read_csv(file).to_numpy()[
                Basic_Stat.velocity.value]
            vel_series = np.multiply(vel_series, 10)
            vel_series = np.floor(vel_series).astype(int)
            for k in range(len(vel_series)):
                index = vel_series[k]
                if (index < 400):
                    vel_hist[index] += 1

        sum = 0
        for j in range(histogram_size):
            sum += vel_hist[j]
        vel_hist = np.divide(vel_hist, sum)
        vel_hist = np.multiply(vel_hist, 100)

        np.savetxt('data/vel-hist/driver' + str(i) + '.csv',
                   vel_hist, fmt='%5f', delimiter=',')


def show_histograms(num_drivers: int, num_show: int = 3, is_rand: bool = False) -> None:
    """Shows histograms of statistics of specific driver(s)

    :param num: int - number of drivers to show (default = 3)
    :param num_driver: int - the total number of drivers
    :param is_rand: bool - if chosen drivers are random, first 'num' if not random (default = False)
    """

    if (num_show > num_drivers):
        print('Number of figures to show is greater than number of drivers present')
        return

    _, axs = plt.subplots(3, num_show)
    plt.gray()

    if (is_rand):
        random.seed(time)
    else:
        random.seed(1)

    if(num_show == 1):
        k = 0
        if(is_rand):
            k = randrange(num_drivers)

        ang_vel_hist = pd.read_csv(
            'data/angular-vel-hist/driver' + str(k) + '.csv', header=None).to_numpy()
        vel_hist = pd.read_csv('data/vel-hist/driver' +
                               str(k) + '.csv', header=None).to_numpy()
        accel_hist = pd.read_csv(
            'data/combined-accel-hist/driver' + str(k) + '.csv', header=None).to_numpy()

        axs[0].plot(vel_hist)
        axs[0].title.set_text('driver ' + str(k) + ' velocity histogram')
        axs[1].plot(accel_hist)
        axs[1].title.set_text('driver ' + str(k) + ' acceleration histogram')
        axs[2].plot(ang_vel_hist)
        axs[2].title.set_text('driver ' + str(k) +
                              ' angular velocity histogram')
    elif(num_show > 1):
        k = 0
        for i in range(num_show):
            if(is_rand):
                k = randrange(num_drivers)
            else:
                k = i

            ang_vel_hist = pd.read_csv(
                'data/angular-vel-hist/driver' + str(k) + '.csv', header=None).to_numpy()
            vel_hist = pd.read_csv('data/vel-hist/driver' +
                                   str(k) + '.csv', header=None).to_numpy()
            accel_hist = pd.read_csv(
                'data/combined-accel-hist/driver' + str(k) + '.csv', header=None).to_numpy()

            axs[0, i].plot(vel_hist)
            axs[0, i].title.set_text(
                'driver ' + str(k) + ' velocity histogram')
            axs[1, i].plot(accel_hist)
            axs[1, i].title.set_text(
                'driver ' + str(k) + ' acceleration histogram')
            axs[2, i].plot(ang_vel_hist)
            axs[2, i].title.set_text(
                'driver ' + str(k) + ' angular velocity histogram')
    else:
        print('Invalid number of drivers to show')

    plt.show()


def show_bar_plots(rand_drivers=False):
    if (rand_drivers):
        random.seed(time)
    else:
        random.seed(1)

    angular_velocity_index = np.around(np.linspace(-2, 2, 401), 1)[:-1]
    velocity_index_1 = np.around(np.linspace(0, 18, 181), 1)[:-1]
    velocity_index_2 = np.around(np.linspace(18, 28, 101), 1)[:-1]
    velocity_index_3 = np.around(np.linspace(28, 40, 121), 1)[:-1]
    accel_index_1 = np.around(np.linspace(0, .7, 8), 1)[:-1]
    accel_index_2 = np.around(np.linspace(.7, 2, 14), 1)[:-1]
    accel_index_3 = np.around(np.linspace(2, 4, 21), 1)[:-1]

    _, axes = plt.subplots(3, 7)

    for i in range(3):
        k = randrange(0, 999)
        ang_vel_hist = pd.read_csv(
            'data/angular-vel-hist/driver' + str(k) + '.csv', header=None).to_numpy()
        vel_hist = pd.read_csv('data/vel-hist/driver' +
                               str(k) + '.csv', header=None).to_numpy()
        accel_hist = pd.read_csv(
            'data/combined-accel-hist/driver' + str(k) + '.csv', header=None).to_numpy()

        velocity_subsection_1 = vel_hist[0:180, 0]
        velocity_subsection_2 = vel_hist[180:280, 0]
        velocity_subsection_3 = vel_hist[280:, 0]

        accel_subsection_1 = accel_hist[0:7, 0]
        accel_subsection_2 = accel_hist[7:20, 0]
        accel_subsection_3 = accel_hist[20:40, 0]

        ang_vel_hist = ang_vel_hist[:, 0]

        axes[i, 0].bar(velocity_index_1, velocity_subsection_1, width=.1)
        axes[i, 1].bar(velocity_index_2, velocity_subsection_2, width=.1)
        axes[i, 2].bar(velocity_index_3, velocity_subsection_3, width=.1)
        axes[i, 3].bar(accel_index_1, accel_subsection_1, width=.1)
        axes[i, 4].bar(accel_index_2, accel_subsection_2, width=.1)
        axes[i, 5].bar(accel_index_3, accel_subsection_3, width=.1)
        axes[i, 6].bar(angular_velocity_index, ang_vel_hist, width=.1)

        axes[i, 0].set_ylim(0, 15)
        axes[i, 1].set_ylim(0, 2)
        axes[i, 2].set_ylim(0, 2)
        axes[i, 3].set_ylim(0, 20)
        axes[i, 4].set_ylim(0, 10)
        axes[i, 5].set_ylim(0, 2)
        axes[i, 6].set_ylim(0, 10)

        axes[i, 0].title.set_text(f'Driver {k} vel low')
        axes[i, 1].title.set_text(f'Driver {k} vel med')
        axes[i, 2].title.set_text(f'Driver {k} vel high')
        axes[i, 3].title.set_text(f'Driver {k} accel low')
        axes[i, 4].title.set_text(f'Driver {k} accel med')
        axes[i, 5].title.set_text(f'Driver {k} accel high')
        axes[i, 6].title.set_text(f'Driver {k} ang-vel')

    plt.show()


if (__name__ == '__main__'):
    # create_ang_vel_histograms('data/Basic Matricies')
    # create_accel_histograms('data/Basic Matricies')
    # create_vel_histograms('data/Basic Matricies')
    # create_combined_accel_histograms('data/Basic Matricies')
    # show_histograms(num_drivers=1000, num_show=8, is_rand=True)
    show_bar_plots(rand_drivers=True)
    pass
