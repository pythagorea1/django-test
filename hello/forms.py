from django import forms

class HelloForm(forms.Form):
    name = forms.CharField(label='name')
    mail = forms.CharField(label='mail')
    age = forms.IntegerField(label='age')
class Options(forms.Form):
    _base = [
        ('BNB', 'BNB'),
        ('BTC', 'BTC'),
        ('ETH', 'ETH'),
        ('USDT', 'USDT')
    ]
    _timespan = [
        ('1m', '1m'),
        ('5m', '5m'),
    ]
    timespan = forms.ChoiceField(label='INTERVAL', choices=_timespan, required=False)
    base = forms.ChoiceField(label='BASE CURRENCY', choices=_base)
    number_of_portfolio = forms.IntegerField(label='NUM OF PORTFOLIO')
