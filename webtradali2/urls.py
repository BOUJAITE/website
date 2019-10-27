from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
	path('/about',views.about),
	path('/ta',views.tech_analysis, name='tech_analysis'),
	path('/pp',views.price_pred, name='price_pred'),
	path('/contact',views.contact),
	path('/summury',views.sum),
	path('/webtradali',views.index ,name='index'),
	]
