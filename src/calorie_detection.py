import json


def get_calories(food_name, json_path="kcal_per_gram_food41.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    food_name = food_name.strip().lower()

    if food_name in data:
        return data[food_name]

    for key in data:
        if key.lower() == food_name:
            return data[key]

    return None


# if __name__ == "__main__":
#     food = input("نام غذا را وارد کنید: ")
#     result = get_calories(food)
#
#     if result is not None:
#         print(f"کالری هر گرم '{food}': {result}")
#     else:
#         print(f"غذای '{food}' در دیتابیس یافت نشد.")