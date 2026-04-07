from django.urls import path
from .views import PatientFoodIntakeRecommenderByDateView, DailyPatientFoodIntakeRecommenderView, WeeklyPatientFoodIntakeRecommenderView, MonthlyPatientFoodIntakeRecommenderView, GeneralPatientFoodIntakeRecommenderView

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    # Patient Daily food intake recommendations
    path("patient/<int:pk>/daily", DailyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/daily/", DailyPatientFoodIntakeRecommenderView.as_view()),

    # Patient food intake recommendations By Date
    path("patient/<int:pk>/daily/<str:date>", PatientFoodIntakeRecommenderByDateView.as_view()),
    path("patient/<int:pk>/daily/<str:date>/", PatientFoodIntakeRecommenderByDateView.as_view()),

    # Patient Weekly food intake recommendations
    path("patient/<int:pk>/weekly", WeeklyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/weekly/", WeeklyPatientFoodIntakeRecommenderView.as_view()),

    # Patient Monthly food intake recommendations
    path("patient/<int:pk>/monthly", MonthlyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/monthly/", MonthlyPatientFoodIntakeRecommenderView.as_view()),


    # General Patient recommendations for macros and micros
    path("patient/<int:pk>/general", GeneralPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/general/", GeneralPatientFoodIntakeRecommenderView.as_view()),   
]

