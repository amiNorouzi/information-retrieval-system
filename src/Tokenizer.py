# coding: utf-8

import codecs
import re

from utils import default_verbs


class Tokenizer:
    """
    >>> tokenizer = Tokenizer()
    >>> tokenizer.tokenize('این جمله (خیلی) پیچیده نیست!!!')
    ['این', 'جمله', '(', 'خیلی', ')', 'پیچیده', 'نیست', '!!!']
    """

    def __init__(self, verbs_file=default_verbs, join_verb_parts=True):
        self._join_verb_parts = join_verb_parts
        self.pattern = re.compile(r'([؟!?]+|\d[\d.:/\\]+|[:.،؛»\])}"«\[({])')  # TODO \d

        if join_verb_parts:
            self.after_verbs = {
                'ام', 'ای', 'است', 'ایم', 'اید', 'اند', 'بودم', 'بودی', 'بود', 'بودیم', 'بودید',
                'بودند', 'باشم', 'باشی', 'باشد', 'باشیم', 'باشید', 'باشند', 'شده_ام', 'شده_ای',
                'شده_است', 'شده_ایم', 'شده_اید', 'شده_اند', 'شده_بودم', 'شده_بودی', 'شده_بود',
                'شده_بودیم', 'شده_بودید', 'شده_بودند', 'شده_باشم', 'شده_باشی', 'شده_باشد', 'شده_باشیم',
                'شده_باشید', 'شده_باشند', 'نشده_ام', 'نشده_ای', 'نشده_است', 'نشده_ایم', 'نشده_اید',
                'نشده_اند', 'نشده_بودم', 'نشده_بودی', 'نشده_بود', 'نشده_بودیم', 'نشده_بودید',
                'نشده_بودند', 'نشده_باشم', 'نشده_باشی', 'نشده_باشد', 'نشده_باشیم', 'نشده_باشید',
                'نشده_باشند', 'شوم', 'شوی', 'شود', 'شویم', 'شوید', 'شوند', 'شدم', 'شدی', 'شد', 'شدیم',
                'شدید', 'شدند', 'نشوم', 'نشوی', 'نشود', 'نشویم', 'نشوید', 'نشوند', 'نشدم', 'نشدی',
                'نشد', 'نشدیم', 'نشدید', 'نشدند', 'می‌شوم', 'می‌شوی', 'می‌شود', 'می‌شویم', 'می‌شوید',
                'می‌شوند', 'می‌شدم', 'می‌شدی', 'می‌شد', 'می‌شدیم', 'می‌شدید', 'می‌شدند', 'نمی‌شوم',
                'نمی‌شوی', 'نمی‌شود', 'نمی‌شویم', 'نمی‌شوید', 'نمی‌شوند', 'نمی‌شدم', 'نمی‌شدی',
                'نمی‌شد', 'نمی‌شدیم', 'نمی‌شدید', 'نمی‌شدند', 'خواهم_شد', 'خواهی_شد', 'خواهد_شد',
                'خواهیم_شد', 'خواهید_شد', 'خواهند_شد', 'نخواهم_شد', 'نخواهی_شد', 'نخواهد_شد',
                'نخواهیم_شد', 'نخواهید_شد', 'نخواهند_شد'
            }

            self.before_verbs = {
                'خواهم', 'خواهی', 'خواهد', 'خواهیم', 'خواهید', 'خواهند', 'نخواهم', 'نخواهی', 'نخواهد',
                'نخواهیم', 'نخواهید', 'نخواهند'
            }

            with codecs.open(verbs_file, encoding='utf8') as verbs_file:
                self.verbs = list(reversed([verb.strip() for verb in verbs_file if verb]))
                self.bons = set([verb.split('#')[0] for verb in self.verbs])
                self.verbe = set([bon + 'ه' for bon in self.bons] + ['ن' + bon + 'ه' for bon in self.bons])

    def tokenize(self, text: str) -> list[str]:
        text = self.pattern.sub(r' \1 ', text.replace('\n', ' ').replace('\t', ' '))

        tokens = [word for word in text.split(' ') if word]
        if self._join_verb_parts:
            tokens = self.join_verb_parts(tokens)
        return tokens

    def join_verb_parts(self, tokens):
        result = ['']
        for token in reversed(tokens):
            if token in self.before_verbs or (result[-1] in self.after_verbs and token in self.verbe):
                result[-1] = token + '_' + result[-1]
            else:
                result.append(token)
        return list(reversed(result[1:]))
