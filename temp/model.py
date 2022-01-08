from wtforms import Form, FloatField, validators, StringField, IntegerField, SelectField, BooleanField, SelectMultipleField, widgets, FieldList, TextAreaField

class InputForm(Form):
    filter = TextAreaField(label='Filter', default="Iulius")
