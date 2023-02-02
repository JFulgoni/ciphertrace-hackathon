import matplotlib.pyplot as plt


def plot_coherence():
    # plotting the points
    plt.plot(x, coherence)

    # naming the x axis
    plt.xlabel('Number of Topics')
    # naming the y axis
    plt.ylabel('Coherence')

    # giving a title to my graph
    plt.title('Number of Topics vs Coherence')

    # function to show the plot
    plt.show()


def plot_perplexity():
    # plotting the points
    plt.plot(x, perplexity)

    # naming the x axis
    plt.xlabel('Number of Topics')
    # naming the y axis
    plt.ylabel('Perplexity')

    # giving a title to my graph
    plt.title('Number of Topics vs Perplexity')

    # function to show the plot
    plt.show()


if __name__ == '__main__':
    # x axis values
    x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    # corresponding y axis values
    perplexity = [-7.5730419437102166, -7.7669197480079, -8.412763076697237, -8.723435592732633, -9.074924504722249,
                  -9.400363692259683, -9.735181937127532, -10.02890887651072, -10.390623980545534, -10.697307186658623,
                  -11.05202904113821]
    coherence = [-1.7607143347377883, -5.271772719916582, -8.777900976281519, -7.076194347551794, -8.19539543124095,
                 -9.011728318539147, -9.325136018164592, -10.624255758629142, -10.438533362362996, -8.909450243696746,
                 -9.499694241452572]

    plot_perplexity()
