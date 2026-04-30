from django.urls import path
from .views import DailyRecommendationsByDummyPatientView, DailyRecommendationsByPatientView, MonthlyRecommendationsByDummyPatientView, MonthlyRecommendationsByPatientView, PatientFoodIntakeRecommenderByDateView, DailyPatientFoodIntakeRecommenderView, WeeklyPatientFoodIntakeRecommenderView, MonthlyPatientFoodIntakeRecommenderView, GeneralPatientFoodIntakeRecommenderView, WeeklyRecommendationsByDummyPatientView, WeeklyRecommendationsByPatientView

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    path("test/patient/<int:pk>/daily/", GeneralPatientFoodIntakeRecommenderView.as_view()),

    # Daily 
    path("nutri-and-food/patient/<int:pk>/daily", DailyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/", DailyRecommendationsByPatientView.as_view()),

    # Weekly
    path("nutri-and-food/patient/<int:pk>/weekly", WeeklyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/weekly/", WeeklyRecommendationsByPatientView.as_view()),

    # Monthly
    path("nutri-and-food/patient/<int:pk>/monthly", MonthlyRecommendationsByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/monthly/", MonthlyRecommendationsByPatientView.as_view()),


    # [PREVIOUS]
    path("nutri-and-food/patient/dummy/daily", DailyRecommendationsByDummyPatientView.as_view()),
    path("nutri-and-food/patient/dummy/daily/", DailyRecommendationsByDummyPatientView.as_view()),

    path("nutri-and-food/patient/dummy/weekly", WeeklyRecommendationsByDummyPatientView.as_view()),
    path("nutri-and-food/patient/dummy/weekly/", WeeklyRecommendationsByDummyPatientView.as_view()),

    path("nutri-and-food/patient/dummy/monthly", MonthlyRecommendationsByDummyPatientView.as_view()),
    path("nutri-and-food/patient/dummy/monthly/", MonthlyRecommendationsByDummyPatientView.as_view()),

    
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
    
]

