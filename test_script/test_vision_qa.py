import json
import requests 
import time 
import os
import io
import base64 
import numpy as np 
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from PIL import Image 
from tqdm import tqdm


class Validator:
    def __init__(self):
        self.api_url = "http://localhost:8000/qa-task"
        self.storage_dir = "../visual-qa"
        self.info_file = os.path.join(self.storage_dir, "info.json")
        self.num_samples = 300
        self.initialize()

    def initialize(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self._generate_and_store_data()

    def _generate_and_store_data(self):
        data = []
        for i in tqdm(range(self.num_samples), desc="Generating dataset"):
            save_path = os.path.join(self.storage_dir, f"sample_{i}.png")
            question, image, answer = self._generate_random_plot_question_and_answer(save_path)
            data.append(
                {
                    "image_path": save_path,
                    "question": question, 
                    "answer": answer
                }
            )
        with open(self.info_file, 'w') as f:
            json.dump(data, f)
    
    def _generate_random_plot_question_and_answer(self, save_path: str = None):
        light_colors = [
            "#FF6F71",  # Light Crimson
            "#FFA07A",  # Light Salmon
            "#FFD700",  # Light Goldenrod
            "#9370DB",  # Light Indigo
            "#FF9F80",  # Light Coral
            "#AFEEEE",  # Light Turquoise
            "#E6A8D7",  # Light Orchid
            "#D2B48C",  # Light Sienna
            "#20B2AA",  # Light Teal
            "#8470FF",  # Light Slate Blue
            "#E5E5FF",  # Light Periwinkle
            "#ADFF2F",  # Light Chartreuse
            "#B0E0E6",  # Light Cadet Blue
            "#FF7F50",  # Light Tomato
            "#9ACD32",  # Light Olive Drab
            "#FF69B4",  # Light Medium Violet Red
            "#FFDAB9",  # Light Peach Puff
            "#98FB98",  # Light Lawn Green
            "#F4A460",  # Light Rosy Brown
            "#E0FFFF"   # Light Pale Turquoise
        ]

        dark_colors = [
            "#8B0000",  # Dark Crimson
            "#E9967A",  # Dark Salmon
            "#B8860B",  # Dark Goldenrod
            "#2F4F4F",  # Dark Indigo
            "#CD5B45",  # Dark Coral
            "#008B8B",  # Dark Turquoise
            "#9932CC",  # Dark Orchid
            "#8B4513",  # Dark Sienna
            "#006666",  # Dark Teal
            "#483D8B",  # Dark Slate Blue
            "#6666FF",  # Dark Periwinkle
            "#556B2F",  # Dark Chartreuse
            "#4682B4",  # Dark Cadet Blue
            "#CD3700",  # Dark Tomato
            "#556B2F",  # Dark Olive Drab
            "#8B008B",  # Dark Medium Violet Red
            "#CD853F",  # Dark Peach Puff
            "#228B22",  # Dark Lawn Green
            "#8B5F65",  # Dark Rosy Brown
            "#5F9EA0"   # Dark Pale Turquoise
        ]
        
        random_terms1 = ['Efficiency', 'Productivity', 'Performance', 'Output', 'Speed', 'Quality', 'Rate', 'Capacity', 'Level', 'Intensity', 'Strength', 'Power']
        random_terms2 = ['Time', 'Date', 'Period', 'Duration', 'Phase', 'Stage', 'Interval', 'Epoch', 'Era', 'Span', 'Cycle', 'Moment']

        random_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd']


        def serialize_image(image: Image.Image) -> str:
            if image:
                # Convert image to byte stream
                if save_path:
                    image.save(save_path)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()

                # Encode byte stream to base64 string
                img_str = base64.b64encode(img_bytes).decode('utf-8')
                return img_str

            return None

        def get_contrasting_colors(num_colors):
            bg_color = random.choice(light_colors + dark_colors)
            if bg_color in light_colors:
                plot_colors = random.sample(dark_colors, num_colors) #sample should return unique colors
            else:
                plot_colors = random.sample(light_colors, num_colors)
            labels_color = random.choice(dark_colors)
            markers = random.sample(random_markers, num_colors)
            return bg_color, plot_colors, markers, labels_color
        
        def generate_evenly_spread_dates(start_date, num_days):
            months = np.random.randint(1, 13)
            days_in_months = months * 30
            dates = [start_date + timedelta(days=int(days_in_months * i / num_days)) for i in range(num_days)]
            return dates
        
        num_days = np.random.randint(25,35)
        start_year = np.random.randint(1970, 2031)
        start_month = np.random.randint(1, 13)
        start_date = datetime(start_year, start_month, 1)
        dates = generate_evenly_spread_dates(start_date, num_days)

        # Determine the number of y-axis terms (1 to 3)
        num_y_terms = random.randint(1, 3)
        
        # Generate random data for each y-axis term
        random_low = np.random.uniform(0, 90)  # Ensuring that low can be up to 90
        random_high = np.random.uniform(random_low + 10, 100)  # Ensuring high is at least 10 more than low
        if random_low > random_high:
            random_low, random_high = random_high, random_low

        data = {f'Data{j+1}': np.random.uniform(low=random_low, high=random_high, size=num_days) for j in range(num_y_terms)}
        
        # Create a DataFrame
        data['Date'] = dates
        df = pd.DataFrame(data)
        
        # Sort data by date to ensure the plot makes sense
        df = df.sort_values(by='Date')
        
        # Randomly select the type of plot
        plot_types = ['line', 'scatter']
        plot_types_with_bar = ['line', 'bar', 'scatter']

        #only do bar graphs if there is one y-term
        if num_y_terms == 1:
            selected_plot_type = random.choice(plot_types_with_bar)
        else:
            selected_plot_type = random.choice(plot_types)
        
        # Randomly select terms for y-axis and time
        y_terms = random.sample(random_terms1, num_y_terms)
        time_term = random.choice(random_terms2)
        
        # Get contrasting colors
        bg_color, plot_colors, markers, labels_color = get_contrasting_colors(num_y_terms)
        
        # Plot the data based on the selected plot type
        plt.figure(figsize=(np.random.randint(8,12), np.random.randint(4,6)))
        plt.gca().set_facecolor(bg_color)
        for j in range(num_y_terms):
            if selected_plot_type == 'line':
                # TODO random choice of markers
                plt.plot(df['Date'], df[f'Data{j+1}'], marker=markers[j], color=plot_colors[j], label=y_terms[j])
            elif selected_plot_type == 'bar':
                plt.bar(df['Date'] + timedelta(days=j*2), df[f'Data{j+1}'], color=plot_colors[j], label=y_terms[j], width=2)
            elif selected_plot_type == 'scatter':
                plt.scatter(df['Date'], df[f'Data{j+1}'], color=plot_colors[j], label=y_terms[j])
        
        plt.title(f'{", ".join(y_terms)} vs {time_term}', color=labels_color)
        plt.xlabel(time_term, color=labels_color)
        plt.ylabel('Value', color=labels_color)
        plt.grid(True, color=plot_colors[0]) #I didnt make this one the label color to make sure it contrasts with the plot background

        # Format the x-axis dates
        date_form = DateFormatter("%m/%d/%Y")
        plt.gca().xaxis.set_major_formatter(date_form)

        plt.xticks(rotation=45, color=labels_color)
        plt.yticks(color=labels_color)
        plt.legend()
        
        plt.tight_layout()
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        # Create an image object from the BytesIO object
        image = Image.open(buf)

        image_base64 = serialize_image(image)

        facts_list = []
        
        # Calculate additional facts
        for j in range(num_y_terms):
            y_col = f'Data{j+1}'
            highest_value = df[y_col].max()
            lowest_value = df[y_col].min()
            diff_high_low = highest_value - lowest_value
            facts_list.extend([(f'What is the highest value of {y_terms[j]}?', f'{highest_value:.2f}'),
                               (f'What is the lowest value of {y_terms[j]}?', f'{lowest_value:.2f}'),
                               (f'What is the difference between the high and low of {y_terms[j]}?', f'{diff_high_low:.2f}'),
            ])
        
        # Calculate dates where y-terms are closest and most different in value
        if num_y_terms > 1:
            for j in range(num_y_terms):
                for k in range(j+1, num_y_terms):
                    diff_series = np.abs(df[f'Data{j+1}'] - df[f'Data{k+1}'])
                    closest_date = df.iloc[diff_series.idxmin()]['Date'].strftime('%m/%d/%Y')
                    largest_diff_date = df.iloc[diff_series.idxmax()]['Date'].strftime('%m/%d/%Y')
                    
                    facts_list.extend([(f'On what date are {y_terms[j]} and {y_terms[k]} closest in value?', closest_date),
                                       (f'On what day is the largest difference between {y_terms[j]} and {y_terms[k]}?', largest_diff_date),
                    ])


        # get a random question
        question, selected_answer = random.choice(facts_list)

        return question, image_base64, selected_answer
    
    def _prepare_payload(self):
        payloads = []
        samples = json.load(open(self.info_file))
        for sample in samples:
            image_path = sample['image_path']
            with open(image_path, 'rb') as img:
                encoded_string = base64.b64encode(img.read())
            payloads.append(
                {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'base64_image': encoded_string.decode('utf-8')
                }
            )
        return payloads

    def api_forward(self, payload):
        start = time.time()
        refined_payload = {
            "prompt": payload['question'],
            "datas": [],
            "files": [{"content": payload['base64_image'], "type": "image"}]
        }
        res = requests.post(self.api_url, json=refined_payload)
        return res, time.time() - start
    
    def __call__(self):
        payloads = self._prepare_payload()
        outputs = []
        for payload in tqdm(payloads, desc="Run testing"):
            res, latency = self.api_forward(payload)
            outputs.append({
                "answer": res.json()['response'],
                "groundtruth": payload['answer'],
                "process_time": latency
            })
        save_testing_path = os.path.join(self.storage_dir, "test_result.json")
        with open(save_testing_path, "w") as f:
            json.dump(outputs, f)
            
if __name__ == "__main__":
    validator = Validator()
    validator()

    