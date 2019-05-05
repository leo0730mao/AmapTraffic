import matplotlib.pyplot as plt
import seaborn as sns


def draw_speed_time_series(data):
    sns.set(color_codes = True)
    plt.figure(0)
    for k in data.keys():
        plt.plot(data[k], label = k)
    plt.xlabel("24h")
    plt.ylabel("average speed(km/h)")
    plt.title('Shanghai weekday')
    plt.xticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144], ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'], rotation=30)
    # plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    plt.legend()
    plt.tight_layout()
    plt.savefig('Shanghai_weekday.jpg')
    plt.close(0)


def draw_one_week_speed(data):
    sns.set(color_codes = True)
    plt.figure(0)
    total_series = []
    date_seires = []
    for k in sorted(data):
        total_series += list(data[k])
        date_seires.append(k)
    date_seires = sorted(date_seires)

    plt.plot(total_series)
    plt.xlabel("one week")
    plt.ylabel("average speed(km/h)")
    plt.title('Shanghai')
    plt.xticks([0, 144, 288, 432, 576, 720, 864], date_seires, rotation = 30)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Shanghai_one_week.jpg')
    plt.close(0)
