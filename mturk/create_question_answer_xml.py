import xmltodict

qs = [('umbrella', 'puddle'),
      ('Avalanche', 'snow-covered house'),
      ('snow-covered car', 'snow-covered house'),
      ('sunscreen', 'sunglasses'),
      ('umbrella', 'rain'),
      ('Sunny sky', 'sunglasses'),
      ('spider', 'scared face'),
      ('spider', 'happy face'),
      ('cold weather', 'shorts'),
      ('a winter coat', 'shorts'),
      ('happy child face', 'cute bunny')]

ans = [
    'z-to-xy',
    'x-to-y',
    'z-to-xy',
    'z-to-xy',
    'y-to-x',
    'x-to-y',
    'x-to-y',
    'x-to-y',
    'x-to-y',
    'z-to-xy',
    'y-to-x',
]

from collections import OrderedDict as od

q_dict = od([
    ('QuestionForm', od([
        ('@xmlns', 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2017-11-06/QuestionForm.xsd'),
        ('Overview', od([
            ('Title', 'Causation or only correlation?'),
            ('FormattedContent ','''
            In this survey, we are interested in finding causal relations between objects in a scene, as opposed to just correlations.
                Your task is to say whether you think intervening on the presence (aka making present / absent) of one object in a scene would change your expectation of seeing the other object present.
                A more detailed explanation is below, and an example is also given in this video: <a href="https://drive.google.com/file/d/1O8NL5R3P9w0DdUEeAlRvVJrFUDIWZaWE/view?usp=sharing" target="_blank">Explanation video</a><br \><br \>
            For example, take a rain cloud, an umbrella and a puddle: the umbrella and the puddle are certainly correlated: if you see the one, you have an increased expectation of seeing the other.
                However, the presence of the umbrella does not <i>cause</i> the presence of the puddle: if I would just put an umbrella in the scene, that wouldn't make puddles appear.
                The correlation happens because of the rain cloud: it <i>does cause</i> the puddle, and it also causes the umbrella: if I 'put' a raining rain cloud in a scene,
                I can expect umbrellas and puddles to start appearing in my scene too.<br \>
                  We write this down compactly as "rain cloud" → "umbrella" and "rain cloud" → "puddle", so "umbrella" ← "rain cloud" → "puddle"<br \>
                Note that we also talk of causation when the presence of one object causes your expectation of seeing the other object to <i>decrease</i>: for example,
                  if I put a rain cloud in a scene, then I expect to see <i>fewer</i> sunglasses in the scene: in this case it is also true that "raincloud → sunglasses".<br \>
                  To show that you understand this, solve this test. You can fill in the actual HIT if you get at most 2 answers wrong.
            ''')
        ])),
        ('Question', [
            od([
                ('QuestionIdentifier', f'q_{i}'),
                ('DisplayName', f'Question {i}'),
                ('IsRequired', 'true'),
                ('QuestionContent', od([
                    ('Text', f'{o1} - {o2}')
                ])),
                ('AnswerSpecification', od([
                    ('SelectionAnswer', od([
                        ('StyleSuggestion', 'radiobutton'),
                        ('Selections', od([
                            ('Selection', [
                                od([
                                    ('SelectionIdentifier', sid),
                                    ('Text', stext),
                                ]) for sid, stext in zip(['x-to-y', 'y-to-x', 'z-to-xy'],
                                                         [f'{o1} \u2192 {o2}', f'{o2} \u2192 {o1}',
                                                          f'{o1} \u2190 something else \u2192 {o2}'])
                            ])
                        ])),
                    ]))
                ]))
            ])
            for i, (o1, o2) in enumerate(qs)
        ]
         ),
    ]))
])
a_dict = od([
    ('AnswerKey', od([
        ('@xmlns', 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/AnswerKey.xsd'),
        ('Question', [
            od([
                ('QuestionIdentifier', f'q_{i}'),
                ('AnswerOption', od([
                    ('SelectionIdentifier', aid),
                    ('AnswerScore', 1)
                ])
                 )
            ])
            for i, aid in enumerate(ans)
        ]),
        ('QualificationValueMapping', od([
            ('PercentageMapping', od([
                ('MaximumSummedScore', 11)
            ]))
        ]))
    ]))
])
with open('questionForm.xml', 'w') as f:
    f.write(xmltodict.unparse(q_dict, pretty=True))
with open('answerKey.xml', 'w') as f:
    f.write(xmltodict.unparse(a_dict, pretty=True))
