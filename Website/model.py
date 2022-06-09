from wtforms import Form, FloatField, validators, StringField, IntegerField, SelectField, BooleanField, SelectMultipleField, widgets, FieldList, TextAreaField

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()

class InputForm(Form):
    def validate(self):
        if not Form.validate(self):
            return False
        result = True
        if self.lAge.data>=self.uAge.data and self.anim.data:
            self.uAge.errors.append('Upper age must be larger than lower age.')
            result = False
        if self.anim.data and type(self.fps.data)!=float:
            self.fps.errors.append('Please set FPS.')
            result = False
        if self.fixedvmax.data and type(self.vmax.data)!=float:
            self.vmax.errors.append('Please set Vmax.')
            result = False
        if len(self.filter.data)==0 and len(self.filterStatus.data)==0:
            self.filter.errors.append('You must filter the data; running the app on the whole corpus might make it crash.')
            result = False
        if self.xmin.data > self.xmax.data or self.ymin.data > self.ymax.data:
            self.zoom.errors.append('Please enter valid coordinates.')
            result = False
        return result

    filter = TextAreaField(label='Filter', default="Iulius")
    filterOperator = SelectField('↑', choices=[('and', 'and'), ('or', 'or')], default="or")
    filterStatus = TextAreaField(label='Category', default="",validators=[])
    filterStatusOperator = SelectField('↑', choices=[('and', 'and'), ('or', 'or')], default="or")
    lAge = IntegerField(label='Age -', default=-100,validators=[validators.InputRequired()])
    weighted = BooleanField('Weighted')
    plotPoints = BooleanField('Points')
    plotKde = BooleanField('Density')
    plotHist2d = BooleanField('Histogram')
    plotClus = BooleanField('Clusters')
    radClus = IntegerField(label='R (km)', default=50)
    #plot = MultiCheckboxField('Plot', choices=[('points', 'Points'), ('kde', 'Density'), ('hist2d', 'Histogram')])
    imageOnly = BooleanField('Image')
    style = SelectField('Map', choices=[('lines', 'Borders only'), ('linesFilled', 'Borders + land'), ('bluemarble', 'Satellite')], default="linesFilled")
    cities = BooleanField('Cities')
    gridSize = IntegerField(label='Regions', default=30,validators=[validators.InputRequired()])
    fixedvmax = BooleanField('Limit')
    vmax = FloatField(label='Limit value', default=10)
    sizeScatter = FloatField(label='Size points', default=10)
    noDates = BooleanField('All times', default=False)
    anim = BooleanField('Animated', default=True)
    uAge = IntegerField(label='Age +', default=200)
    fps = FloatField(label='FPS', default=15)
    exclureRome = BooleanField('Exclure Rome', default=False)
    datasetApprox = BooleanField('Datation approx.')
    zoom = BooleanField('Zoom')
    xmin = FloatField(label='Xmin', default=0)
    xmax = FloatField(label='Xmax', default=6698000)
    ymin = FloatField(label='Ymin', default=0)
    ymax = FloatField(label='Ymax', default=4896000)
