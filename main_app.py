from flask import Flask, render_template, request,redirect, url_for
import joblib
import pandas as pd


app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

df = pd.read_csv('Cleaned_Indian_Food_Dataset.csv')

def predict_recipe_names(ingredients_input):
    new_ingredients = [ingredient.strip() for ingredient in ingredients_input.split(',')]
    new_ingredients_vectorized = vectorizer.transform(new_ingredients)
    predicted_recipe_names = model.predict(new_ingredients_vectorized)
    return predicted_recipe_names


def get_recipe_info(recipe_name):
    recipe_info = df[df['TranslatedRecipeName'] == recipe_name]
    return recipe_info.iloc[0]

def find_matching_recipes(recipe_name):
    exclusion_list = ["recipe", "masala"]
    if any(exclude.lower() in recipe_name.lower() for exclude in exclusion_list):
        return None
    # Find recipes containing the input recipe name in the recipe name or ingredients list
    matching_recipes = df[df['TranslatedRecipeName'].str.contains(recipe_name, case=False)]
    return matching_recipes

@app.route('/generate', methods=['POST'])
def generate():
    ingredients_input = request.form['ingredients']
    predicted_recipe_names = predict_recipe_names(ingredients_input)

    recipes = []
    for recipe_name in predicted_recipe_names:
        recipe_info = get_recipe_info(recipe_name)
        recipe_dict = {
            'name': recipe_name,
            'url': recipe_info['URL'],
            'translated_instructions': recipe_info['TranslatedInstructions'].split('.')[: -1],
            'total_time': recipe_info['TotalTimeInMins'],
            'ingredient_count': recipe_info['Ingredient-count'],
            'ingredients': recipe_info['Cleaned-Ingredients'],
            'image_url': recipe_info['image-url'],
        }
        recipes.append(recipe_dict)

    return render_template('recipe_searched.html', recipes=recipes)
@app.route('/search', methods=['POST'])
def search():
    recipes = []
    ingredients_input = request.form['recipe_input']
    matching_recipe_names = find_matching_recipes(ingredients_input)
    if matching_recipe_names is not None and not matching_recipe_names.empty:
        for idx,recipe_info in matching_recipe_names.iterrows():
            
            recipe_dict = {
                'name': recipe_info['TranslatedRecipeName'],
                'url': recipe_info['URL'],
                'translated_instructions': recipe_info['TranslatedInstructions'].split('.')[: -1],
                'total_time': recipe_info['TotalTimeInMins'],
                'ingredient_count': recipe_info['Ingredient-count'],
                'ingredients': recipe_info['Cleaned-Ingredients'],
                'image_url': recipe_info['image-url'],
            }
            recipes.append(recipe_dict)
        recipes.append(recipe_dict)

    return render_template('recipe_searched.html', recipes=recipes)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/receipe')
def receipe():
    return render_template('receipe-post.html')

@app.route('/redirect_receipe')
def redirect_receipe():
    return redirect(url_for('receipe'))

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/redirect_contact')
def redirect_contact():
    return redirect(url_for('contact'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/redirect_about')
def redirect_about():
    return redirect(url_for('about'))

@app.route('/redirect_index')
def redirect_index():
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
