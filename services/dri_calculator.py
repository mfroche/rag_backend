def get_dri_water(sex):
    if sex == 'male':
        dri_water = 3.7
    elif sex == 'female':
        dri_water = 2.7
        
    return dri_water


def get_estimated_energy_requirements(age, sex, physical_activity_level, height, weight):   
    eer = 0.0
    if sex == 'male':
        if physical_activity_level == 'inactive':
            eer = 753.07 - (10.83 * age) + (6.50 * height) + (14.10 * weight)
        elif physical_activity_level == 'low_active':
            eer = 581.47 - (10.83 * age) + (8.30 * height) + (14.94 * weight)
        elif physical_activity_level == 'active':
            eer = 1004.82 - (10.83 * age) + (6.52 * height) + (15.91 * weight)
        elif physical_activity_level == 'very_active':
            eer = 517.88 - (10.83 * age) + (15.61 * height) + (19.11 * weight)
        else:
            raise ValueError("Invalid physical activity level")
    elif sex == 'female':
        if physical_activity_level == 'inactive':
            eer = 584.90 - (7.01 * age) + (5.72 * height) + (11.71 * weight)
        elif physical_activity_level == 'low_active':
            eer = 575.77 - (7.01 * age) + (6.60 * height) + (12.14 * weight)
        elif physical_activity_level == 'active':
            eer = 710.25 - (7.01 * age) + (6.54 * height) + (12.34 * weight)
        elif physical_activity_level == 'very_active':
            eer = 511.83 - (7.01 * age) + (9.07 * height) + (12.56 * weight)
    
    return eer



def get_dri_fats(age, sex, physical_activity_level, height, weight):
    eer = get_estimated_energy_requirements(age, sex, physical_activity_level, height, weight)
    min = round((0.20 * eer)/9, 2)
    max = round((0.35 * eer)/9, 2)

    return min, max


def get_dri_carbohydrates(age, sex, physical_activity_level, height, weight):
    eer = get_estimated_energy_requirements(age, sex, physical_activity_level, height, weight)
    min = round((0.45 * eer)/4, 2)
    max = round((0.65 * eer)/4, 2)

    return min, max


def get_dri_protein(age, sex, physical_activity_level, height, weight):
    eer = get_estimated_energy_requirements(age, sex, physical_activity_level, height, weight)
    dri_protein = 0.8 * weight
    min = round((0.10 * eer)/4, 2)
    max = round((0.35 * eer)/4, 2)

    return dri_protein, min, max


def get_dri_fiber(age, sex, physical_activity_level, height, weight):
    eer = get_estimated_energy_requirements(age, sex, physical_activity_level, height, weight)
    dri_fiber = 14 * (eer / 1000)

    return dri_fiber