import re

# TODO Define other Regex here
mails = [
    (re.compile(r'(\w+)@(\w+)\.(\w+)'), '$1@$2.$3')
]

socials = [
    (re.compile(r'https://(\w+)/([^/]+)'), '$1:$2')
]

tels = [
    (re.compile(r'(\d\d)-(\d\d)-(\d\d)-(\d\d)'), '(0$1) $2 $3 $4')
]
