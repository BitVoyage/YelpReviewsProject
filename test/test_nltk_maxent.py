import numpy
import scipy
import nltk

print 'Numpy Version:', numpy.__version__
print 'Scipy Version:', scipy.__version__
print 'NLTK Version:', nltk.__version__

print nltk.usage(nltk.ClassifierI)

train = [
     (dict(a=1, b=1, c=1), 'y'),
     (dict(a=1, b=1, c=1), 'x'),
     (dict(a=1, b=1, c=0), 'y'),
     (dict(a=0, b=1, c=1), 'x'),
     (dict(a=0, b=1, c=1), 'y'),
     (dict(a=0, b=0, c=1), 'y'),
     (dict(a=0, b=1, c=0), 'x'),
     (dict(a=0, b=0, c=0), 'x'),
     (dict(a=0, b=1, c=1), 'y')
     ]
test = [
     (dict(a=1, b=0, c=1)),  # unseen
     (dict(a=1, b=0, c=0)),  # unseen
     (dict(a=0, b=1, c=1)),  # seen 3 times, labels=y,y,x
     (dict(a=0, b=1, c=0))  # seen 1 time, label=x
     ]


def test_maxent():
    classifiers = {}
    for algorithm in nltk.classify.MaxentClassifier.ALGORITHMS:
        if algorithm == 'MEGAM' or algorithm == 'TADM':
            print '(skipping %s)' % algorithm
        else:
            try:
                classifiers[algorithm] = nltk.MaxentClassifier.train(
                     train, algorithm, trace=0, max_iter=1000)
            except Exception, e:
                classifiers[algorithm] = e
    print ' '*11 + ''.join(['        test[%s]' % i for i in range(len(test))])
    print ' '*11 + '      p(x), P(y)'*len(test)
    print '-'*(11 + 15*len(test))
    for algorithm, classifier in classifiers.items():
         print ' %11s' % algorithm,
         if isinstance(classifier, Exception):
             print 'Error: %r' % classifier
             continue
         for featureset in test:
              pdist = classifier.prob_classify(featureset)
              print '%8.2f%6.2f' % (pdist.prob('x'), pdist.prob('y')),
         print


def test_simple():
    train_s = [
        (dict(a=(-1, -0.5), b=(-1, -1), c=(-1, -3), d=(-1, -1), e=3.0), '-1'),
        (dict(a=(1, 0.5), b=(-0.5, 1), c=(1, 3), d=(1, 1), e=4.0), '1'),
        (dict(a=(0.7, 1), b=(0.3, 1), c=(0.6, 3), d=(1, 1), e=4.5), '1'),
        (dict(a=(-1, -0.5), b=(-2, -1), c=(-2, -3), d=(-1, -1), e=2.0), '-1')
    ]
    test_s = [
         (dict(a=(-1, -0.5), b=(-1, -1), c=(-0.7, -3), d=(-0.9, 1), e=3.2)),
         (dict(a=(0.8, 0.5), b=(-0.2, 1), c=(1, 2), d=(1, 1), e=4.0))
    ]
    classifier = nltk.classify.MaxentClassifier.train(train_s, 'IIS', trace=0, max_iter=1000)
    for featureset in test_s:
        pdist = classifier.prob_classify(featureset)
        print pdist.prob('1'), pdist.prob('-1')


def test_bag_words():
    train_s = [
        (dict(a=1, b=3, c=3, d=2, e=3), '-1'),
        (dict(a=3, f=3, g=3, r=3), '1'),
        (dict(a=1, b=3, c=3, d=2, e=3, h=3), '1'),
        (dict(a=1, b=3, c=3, d=4), '-1')
            ]
    test_s = [
        (dict(a=1, b=3, c=3, r=2, e=1)),
        (dict(a=1, f=3, g=3, r=2, d=1))
    ]
    classifier = nltk.classify.MaxentClassifier.train(train_s, 'IIS', trace=0, max_iter=1000)
    for featureset in test_s:
        pdist = classifier.prob_classify(featureset)
        print pdist.prob('1'), pdist.prob('-1')

test_bag_words()

