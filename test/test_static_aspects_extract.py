from opinion_mining.static_aspects_extract import StaticAspectsExtract as SAE


def test_get_features():
    SAE.get_features(1)


def test_train():
    SAE.train(1)


def test_test():
    SAE.test(1)

if __name__ == "__main__":
    # TODO encoding utf-8 in a more elegant way
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # test_get_features()
    # test_train()
    test_test()

