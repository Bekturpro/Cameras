# Начало
import json
import csv
import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import linprog
import time

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("ВНИМАНИЕ: matplotlib не установлен. Графики не будут отображаться.")
    print("Установите: pip install matplotlib numpy")

class ProductSample:
    
    def __init__(self, name, product_type):
        self.name = name
        self.product_type = product_type
        self.characteristics = {}
        self.technical_level = 0
    
    def add_characteristic(self, char_name, value, weight=1.0):
        self.characteristics[char_name] = {
            'value': value,
            'weight': weight
        }
    
    def get_characteristic_value(self, char_name):
        if char_name in self.characteristics:
            return self.characteristics[char_name]['value']
        return None

class TechnicalLevelCalculator:
    
    def __init__(self):
        self.base_values = {
            'IP-камера': {
                'Разрешение (MP)': 4.0,
                'Частота кадров (fps)': 30,
                'Чувствительность (lux)': 0.01,
                'Угол обзора (град)': 90,
                'Сжатие видео (коэффициент)': 0.5,
                'Сеть (Mbps)': 100
            },
            'Аналоговая камера': {
                'Разрешение (TVL)': 700,
                'Частота кадров (fps)': 25,
                'Чувствительность (lux)': 0.1,
                'Угол обзора (град)': 90,
                'Качество сигнала (коэфф)': 0.9
            },
            'Видеорегистратор': {
                'Количество каналов': 16,
                'Разрешение записи (MP)': 4.0,
                'Частота кадров (fps)': 30,
                'Объем памяти (TB)': 4.0,
                'Сжатие видео (коэффициент)': 0.5,
                'Сеть (Mbps)': 1000
            }
        }
        
        self.weights = {
            'IP-камера': {
                'Разрешение (MP)': 1.5,
                'Частота кадров (fps)': 1.0,
                'Чувствительность (lux)': 1.2,
                'Угол обзора (град)': 0.8,
                'Сжатие видео (коэффициент)': 1.0,
                'Сеть (Mbps)': 0.5
            },
            'Аналоговая камера': {
                'Разрешение (TVL)': 1.5,
                'Частота кадров (fps)': 1.0,
                'Чувствительность (lux)': 1.2,
                'Угол обзора (град)': 0.8,
                'Качество сигнала (коэфф)': 1.0
            },
            'Видеорегистратор': {
                'Количество каналов': 1.5,
                'Разрешение записи (MP)': 1.2,
                'Частота кадров (fps)': 1.0,
                'Объем памяти (TB)': 1.3,
                'Сжатие видео (коэффициент)': 1.0,
                'Сеть (Mbps)': 0.5
            }
        }
        
        self.lower_is_better = ['Чувствительность (lux)', 'Сжатие видео (коэффициент)']
    
    def calculate_relative_characteristics(self, sample):
        relative = {}
        base = self.base_values.get(sample.product_type, {})
        
        for char_name, char_data in sample.characteristics.items():
            value = char_data['value']
            base_value = base.get(char_name, 1.0)
            
            if base_value != 0:
                if char_name in self.lower_is_better:
                    rel_value = base_value / value if value != 0 else 0
                else:
                    rel_value = value / base_value
            else:
                rel_value = 0
            
            relative[char_name] = round(rel_value, 3)
        
        return relative
    
    def calculate_technical_level(self, sample):
        relative = self.calculate_relative_characteristics(sample)
        weights = self.weights.get(sample.product_type, {})
        
        total_weight = 0
        weighted_sum = 0
        
        for char_name, rel_value in relative.items():
            weight = weights.get(char_name, 1.0)
            weighted_sum += rel_value * weight
            total_weight += weight
        
        if total_weight > 0:
            technical_level = weighted_sum / total_weight
        else:
            technical_level = 0
        
        sample.technical_level = round(technical_level, 3)
        return sample.technical_level

class DataManager:
    
    @staticmethod
    def save_to_json(samples, filename='products.json'):
        data = []
        for sample in samples:
            data.append({
                'name': sample.name,
                'product_type': sample.product_type,
                'characteristics': sample.characteristics,
                'technical_level': sample.technical_level
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в {filename}")
    
    @staticmethod
    def load_from_json(filename='products.json'):
        samples = []
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return samples
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            sample = ProductSample(item['name'], item['product_type'])
            sample.characteristics = item['characteristics']
            sample.technical_level = item.get('technical_level', 0)
            samples.append(sample)
        
        print(f"Загружено {len(samples)} образцов из {filename}")
        return samples
    
    @staticmethod
    def save_to_csv(samples, filename='products.csv'):
        if not samples:
            print("Нет данных для сохранения")
            return
        
        all_chars = set()
        for sample in samples:
            all_chars.update(sample.characteristics.keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Название', 'Тип', 'Техн. уровень'] + list(all_chars)
            writer.writerow(header)
            
            for sample in samples:
                row = [sample.name, sample.product_type, sample.technical_level]
                for char in all_chars:
                    value = sample.characteristics.get(char, {}).get('value', '')
                    row.append(value)
                writer.writerow(row)
        
        print(f"Данные сохранены в {filename}")

class Visualizer:
    
    @staticmethod
    def plot_radial_chart(samples, calculator):
        if not HAS_MATPLOTLIB or not samples:
            print("Нет данных для построения графика")
            return

        available_types = list(set(s.product_type for s in samples))

        if len(available_types) > 1:
            print("\n" + "="*50)
            print("ВЫБОР ТИПА ПРОДУКЦИИ ДЛЯ РАДИАЛЬНОЙ ДИАГРАММЫ")
            print("="*50)
            for i, p_type in enumerate(available_types, 1):
                print(f"{i}. {p_type}")
            print("="*50)
            
            while True:
                try:
                    choice = int(input("Введите номер типа (1-{}): ".format(len(available_types))))
                    if 1 <= choice <= len(available_types):
                        selected_type = available_types[choice - 1]
                        break
                    print("Неверный номер. Попробуйте снова.")
                except ValueError:
                    print("Ошибка: введите число.")
        else:
            selected_type = available_types[0]
            print(f"\nАвтоматически выбран тип: {selected_type}")

        type_samples = [s for s in samples if s.product_type == selected_type]
        if not type_samples:
            print("Нет образцов выбранного типа.")
            return

        all_chars = set()
        for s in type_samples:
            all_chars.update(s.characteristics.keys())
        labels = list(all_chars)
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
        colors = plt.cm.tab10(np.linspace(0, 1, len(type_samples)))

        for i, sample in enumerate(type_samples):
            rel = calculator.calculate_relative_characteristics(sample)
            vals = [rel.get(l, 0) for l in labels]
  
            ax.plot(angles + [angles[0]], vals + [vals[0]], 'o-', linewidth=2,
                    label=sample.name, color=colors[i], alpha=0.8)
            ax.fill(angles + [angles[0]], vals + [vals[0]], alpha=0.15, color=colors[i])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), labels)

        all_vals = []
        for s in type_samples:
            all_vals.extend(calculator.calculate_relative_characteristics(s).values())
        max_v = max(all_vals) if all_vals else 2
        ax.set_ylim(0, max_v * 1.2)

        ax.set_title(f'Относительные характеристики: {selected_type}', pad=20, fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        safe_name = selected_type.replace(' ', '_').replace('/', '_')
        plt.savefig(f'radial_{safe_name}.png', dpi=300, bbox_inches='tight')
        print(f" Диаграмма сохранена: radial_{safe_name}.png")
        plt.show()
    
    @staticmethod
    def plot_bar_chart(samples):
        if not HAS_MATPLOTLIB:
            print("Невозможно построить график: matplotlib не установлен")
            return
        
        if not samples:
            print("Нет образцов для отображения")
            return
        
        sorted_samples = sorted(samples, key=lambda x: x.technical_level, reverse=True)
        
        names = [s.name for s in sorted_samples]
        levels = [s.technical_level for s in sorted_samples]
        types = [s.product_type for s in sorted_samples]
        
        colors = []
        type_colors = {'IP-камера': '#3498db', 'Аналоговая камера': '#e74c3c', 
                      'Видеорегистратор': '#2ecc71'}
        for t in types:
            colors.append(type_colors.get(t, '#95a5a6'))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(names)), levels, color=colors, alpha=0.7)
        
        for i, (bar, level) in enumerate(zip(bars, levels)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{level:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Образцы продукции', fontsize=11)
        ax.set_ylabel('Технический уровень', fontsize=11)
        ax.set_title('Технический уровень продукции (по убыванию)', fontsize=12, pad=15)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                      markerfacecolor=color, markersize=10, label=ptype)
                          for ptype, color in type_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('technical_level_bar.png', dpi=300, bbox_inches='tight')
        print("Столбчатая диаграмма сохранена в technical_level_bar.png")
        plt.show()

def get_product_type():
    print("\n" + "="*50)
    print("Выберите тип продукции:")
    print("1. IP-камера")
    print("2. Аналоговая камера")
    print("3. Видеорегистратор")
    print("0. Вернуться в главное меню")
    print("="*50)
    
    while True:
        choice = input("Введите номер (0-3): ").strip()
        if choice == '0':
            return None
        elif choice == '1':
            return 'IP-камера'
        elif choice == '2':
            return 'Аналоговая камера'
        elif choice == '3':
            return 'Видеорегистратор'
        else:
            print("Неверный ввод. Попробуйте снова.")

def get_characteristics(product_type):
    characteristics = {}
    
    print(f"\nВведите характеристики для '{product_type}':")
    print("(нажмите Enter для использования значения по умолчанию)")
    print("(введите '0' для возврата в главное меню)")
    
    if product_type == 'IP-камера':
        chars = {
            'Разрешение (MP)': '4',
            'Частота кадров (fps)': '30',
            'Чувствительность (lux)': '0.01',
            'Угол обзора (град)': '90',
            'Сжатие видео (коэффициент)': '0.5',
            'Сеть (Mbps)': '100'
        }
    elif product_type == 'Аналоговая камера':
        chars = {
            'Разрешение (TVL)': '700',
            'Частота кадров (fps)': '25',
            'Чувствительность (lux)': '0.1',
            'Угол обзора (град)': '90',
            'Качество сигнала (коэфф)': '0.9'
        }
    else:
        chars = {
            'Количество каналов': '16',
            'Разрешение записи (MP)': '4',
            'Частота кадров (fps)': '30',
            'Объем памяти (TB)': '4',
            'Сжатие видео (коэффициент)': '0.5',
            'Сеть (Mbps)': '1000'
        }
    
    for char_name, default_value in chars.items():
        while True:
            try:
                value = input(f"  {char_name} [{default_value}]: ").strip()
                if value == '0':
                    return None
                if value == '':
                    value = default_value
                value = float(value)
                characteristics[char_name] = value
                break
            except ValueError:
                print("  Ошибка! Введите числовое значение или '0' для выхода.")
    
    return characteristics

def add_sample(samples, calculator):
    print("\n" + "="*50)
    print("ДОБАВЛЕНИЕ НОВОГО ОБРАЗЦА")
    print("="*50)
    
    name = input("Введите название образца: ").strip()
    if not name:
        name = f"Образец_{len(samples) + 1}"
    
    product_type = get_product_type()
    if product_type is None:
        print("Возврат в главное меню...")
        return samples
    
    sample = ProductSample(name, product_type)
    
    characteristics = get_characteristics(product_type)
    if characteristics is None:
        print("Возврат в главное меню...")
        return samples
    
    weights = calculator.weights.get(product_type, {})
    for char_name, value in characteristics.items():
        weight = weights.get(char_name, 1.0)
        sample.add_characteristic(char_name, value, weight)
    
    calculator.calculate_technical_level(sample)
    
    samples.append(sample)
    
    print(f"\n Образец '{name}' добавлен!")
    print(f"  Технический уровень: {sample.technical_level:.3f}")
    
    return samples

def load_sample_from_file(samples, calculator):
    print("\n" + "="*50)
    print("ЗАГРУЗКА ИЗ ФАЙЛА")
    print("="*50)
    print("1. Загрузить из JSON")
    print("2. Загрузить из CSV")
    print("0. Вернуться в главное меню")
    
    choice = input("Выберите формат (0-2): ").strip()
    
    if choice == '0':
        return samples
    elif choice == '1':
        filename = input("Введите имя файла JSON [products.json]: ").strip()
        if not filename:
            filename = 'products.json'
        loaded = DataManager.load_from_json(filename)
        samples.extend(loaded)
    elif choice == '2':
        filename = input("Введите имя файла CSV [products.csv]: ").strip()
        if not filename:
            filename = 'products.csv'
        
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден.")
            return samples
            
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            char_names = header[3:]
            
            for row in reader:
                if not row: continue
                sample = ProductSample(row[0], row[1])
                sample.technical_level = float(row[2]) if row[2] else 0.0
                
                weights = calculator.weights.get(row[1], {})
                for i, char in enumerate(char_names):
                    if 3 + i < len(row) and row[3 + i].strip():
                        sample.add_characteristic(char, float(row[3 + i]), weights.get(char, 1.0))
                samples.append(sample)
        print(f"Загружено {len(samples)} образцов из {filename}")
    else:
        print("Неверный выбор")
    
    return samples

def show_all_samples(samples):
    print("\n" + "="*70)
    print("СПИСОК ВСЕХ ОБРАЗЦОВ")
    print("="*70)
    
    if not samples:
        print("Нет добавленных образцов")
        return
    
    print(f"{'№':<3} {'Название':<25} {'Тип':<20} {'Техн. уровень':<15}")
    print("-"*70)
    
    for i, sample in enumerate(samples, 1):
        print(f"{i:<3} {sample.name:<25} {sample.product_type:<20} {sample.technical_level:<15.3f}")
        
        for char_name, char_data in sample.characteristics.items():
            print(f"    {char_name}: {char_data['value']}")
    
    print("="*70)

class ForecastModule:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.model_type = ''
        self.poly_transformer = None

    def load_cctv_data(self, filename='cctv.csv'):
        try:
            self.df = pd.read_csv(filename)
            features = ['avg_system_price', 'camera_resolution', 'is_ai_analytics', 'storage_capacity', 'night_vision_range']
            target = 'monthly_cctv_sales'
            
            missing_cols = [col for col in features + [target] if col not in self.df.columns]
            if missing_cols:
                print(f"Ошибка: в файле отсутствуют колонки: {missing_cols}")
                return False
            
            self.X = self.df[features]
            self.y = self.df[target]
            
            print(f"\nДанные успешно загружены из {filename}")
            print(f"  Найдено {len(self.df)} записей.")
            print(f"  Целевой показатель: {target}")
            print(f"  Признаки: {', '.join(features)}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False

    def analyze_and_choose_model(self):
        if self.X is None or self.y is None:
            print("Нет данных для обучения. Сначала загрузите CSV.")
            return
            
        print("\n" + "= "*60)
        print("1. АНАЛИЗ ИСХОДНЫХ ДАННЫХ И КОРРЕЛЯЦИЙ")
        print("= "*60)
        print("Гипотеза: признаки с высокой абсолютной корреляцией сильнее влияют на продажи.")
        correlations = {}
        for col in self.X.columns:
            corr = self.X[col].corr(self.y)
            correlations[col] = abs(corr)
            direction = "прямая" if corr > 0 else "обратная"
            print(f"  • {col}: {corr:.4f} ({direction} связь)")
        
        print("\n" + "= "*60)
        print("2. ОБУЧЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
        print("= "*60)
        
        lin_reg = LinearRegression()
        lin_reg.fit(self.X, self.y)
        y_pred_lin = lin_reg.predict(self.X)
        
        r2_lin = r2_score(self.y, y_pred_lin)
        mse_lin = mean_squared_error(self.y, y_pred_lin)
        mae_lin = mean_absolute_error(self.y, y_pred_lin)
        rmse_lin = np.sqrt(mse_lin)
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.X)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, self.y)
        y_pred_poly = poly_reg.predict(X_poly)
        
        r2_poly = r2_score(self.y, y_pred_poly)
        mse_poly = mean_squared_error(self.y, y_pred_poly)
        mae_poly = mean_absolute_error(self.y, y_pred_poly)
        rmse_poly = np.sqrt(mse_poly)
        
        print("\nМетрики Линейной модели:")
        print(f"  R² = {r2_lin:.4f}, MSE = {mse_lin:.2f}, MAE = {mae_lin:.2f}, RMSE = {rmse_lin:.2f}")
        print("\nМетрики Полиномиальной модели (степень 2):")
        print(f"  R² = {r2_poly:.4f}, MSE = {mse_poly:.2f}, MAE = {mae_poly:.2f}, RMSE = {rmse_poly:.2f}")
        
        print("\n" + "-"*60)
        print("3. ВЫБОР НАИЛУЧШЕЙ МОДЕЛИ")
        print("-"*60)
        
        r2_improvement = r2_poly - r2_lin
        if r2_improvement > 0.05 and mse_poly < mse_lin and mae_poly < mae_lin:
            self.model = poly_reg
            self.model_type = 'poly'
            self.poly_transformer = poly
            print("✓ Выбрана ПОЛИНОМИАЛЬНАЯ модель")
            print(f"  Прирост R²: +{r2_improvement:.4f}")
            print(f"  Риск переобучения: низкий (ошибки аппроксимации уменьшились)")
        else:
            self.model = lin_reg
            self.model_type = 'linear'
            print("✓ Выбрана ЛИНЕЙНАЯ модель")
            print("  Полиномиальная модель не дала значимого улучшения или показала признаки переобучения.")
        
        print("\n" + "-"*60)
        print("УРАВНЕНИЕ РЕГРЕССИИ В ЯВНОМ ВИДЕ")
        print("-"*60)
        
        if self.model_type == 'linear':

            terms = []
            for i, col in enumerate(self.X.columns):
                coef = lin_reg.coef_[i]
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}*{col}")
            equation = "y = " + " ".join(terms) + f" + ({lin_reg.intercept_:.4f})"
            print(equation)
            
            print("\nКоэффициенты модели:")
            for i, col in enumerate(self.X.columns):
                print(f"  a{i+1} ({col}) = {lin_reg.coef_[i]:.4f}")
            print(f"  b (свободный член) = {lin_reg.intercept_:.4f}")
        else:
            print("Полиномиальная модель (степень 2) содержит перекрёстные члены и квадраты признаков.")
            print("Уравнение имеет вид: y = Σ(aᵢⱼ·xᵢ·xⱼ) + Σ(bᵢ·xᵢ) + c")
            print("\nКоэффициенты при линейных членах:")
            feature_names = poly.get_feature_names_out(self.X.columns)
            for i, name in enumerate(feature_names):
                if poly_reg.coef_[i] != 0:
                    print(f"  {name}: {poly_reg.coef_[i]:.4f}")
            print(f"  Свободный член: {poly_reg.intercept_:.4f}")
        
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(self.y, y_pred_lin if self.model_type == 'linear' else y_pred_poly, 
                       color='blue', alpha=0.6, label='Прогноз модели')
            min_val = min(self.y.min(), (y_pred_lin if self.model_type == 'linear' else y_pred_poly).min())
            max_val = max(self.y.max(), (y_pred_lin if self.model_type == 'linear' else y_pred_poly).max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное совпадение')
            ax.set_xlabel('Реальные продажи')
            ax.set_ylabel('Прогнозируемые продажи')
            ax.set_title(f'Множественная регрессия: Прогноз vs Факт (R²={r2_lin if self.model_type=="linear" else r2_poly:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig('multiple_regression_fit.png', dpi=300, bbox_inches='tight')
            print("\n✓ График соответствия модели сохранен в multiple_regression_fit.png")
            plt.show()
        
        return True

    def predict_manual(self):
        if self.model is None:
            print("Сначала постройте и выберите модель (пункт 1)")
            return

        print("\n" + "= "*60)
        print("ПРОГНОЗ ПРИ РУЧНОМ ВВОДЕ ВСЕХ ПРИЗНАКОВ")
        print("= "*60)
        
        input_data = {}
        print("Введите значения для всех признаков:")
        for col in self.X.columns:
            min_v = self.X[col].min()
            max_v = self.X[col].max()
            while True:
                try:
                    val = float(input(f"  • {col} (диапазон {min_v:.2f} - {max_v:.2f}): "))
                    input_data[col] = val
                    break
                except ValueError:
                    print("    Ошибка: введите корректное числовое значение.")
        
        df_input = pd.DataFrame([input_data], columns=self.X.columns)
        
        if self.model_type == 'linear':
            prediction = self.model.predict(df_input)[0]
        else:
            prediction = self.model.predict(self.poly_transformer.transform(df_input))[0]
            
        print(f"\n✓ Прогнозируемый объем продаж при заданных параметрах: {prediction:.2f} ед.")

    def predict_interval(self):
        if self.X is None or self.y is None:
            print("Нет данных. Сначала загрузите CSV.")
            return

        print("\n" + "= "*60)
        print("ИНТЕРВАЛЬНЫЙ ПРОГНОЗ")
        print("= "*60)

        features_list = list(self.X.columns)
        print("Выберите признак для интервального прогноза:")
        for i, col in enumerate(features_list, 1):
            print(f"  {i}. {col}")
            
        while True:
            try:
                feat_choice = int(input(f"\nВведите номер признака (1-{len(features_list)}): "))
                if 1 <= feat_choice <= len(features_list):
                    chosen_feature = features_list[feat_choice - 1]
                    break
                print("Неверный номер.")
            except ValueError:
                print("Ошибка: введите число.")

        x_data = self.X[chosen_feature].values
        y_data = self.y.values
        X_single_2d = x_data.reshape(-1, 1)

        lin_reg = LinearRegression()
        lin_reg.fit(X_single_2d, y_data)
        y_pred_lin = lin_reg.predict(X_single_2d)
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_single_2d)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y_data)
        y_pred_poly = poly_reg.predict(X_poly)

        r2_lin = r2_score(y_data, y_pred_lin)
        mse_lin = mean_squared_error(y_data, y_pred_lin)
        r2_poly = r2_score(y_data, y_pred_poly)
        mse_poly = mean_squared_error(y_data, y_pred_poly)

        if (r2_poly - r2_lin) > 0.05 and mse_poly < mse_lin:
            model = poly_reg
            transformer = poly
            model_type = 'poly'
            print(f"\n✓ Для признака '{chosen_feature}' выбрана ПОЛИНОМИАЛЬНАЯ модель (R²={r2_poly:.4f})")
        else:
            model = lin_reg
            transformer = None
            model_type = 'linear'
            print(f"\n✓ Для признака '{chosen_feature}' выбрана ЛИНЕЙНАЯ модель (R²={r2_lin:.4f})")

        print("\nДопустимый диапазон значений:")
        min_v = self.X[chosen_feature].min()
        max_v = self.X[chosen_feature].max()
        print(f"  • {chosen_feature}: от {min_v:.2f} до {max_v:.2f}")

        try:
            start = float(input("  Начальное значение интервала: "))
            end = float(input("  Конечное значение интервала: "))
            steps = int(input("  Количество точек: "))
        except ValueError:
            print("Ошибка: введите корректные числовые значения.")
            return

        X_interval = np.linspace(start, end, steps).reshape(-1, 1)

        if model_type == 'linear':
            y_interval = model.predict(X_interval)
        else:
            y_interval = model.predict(transformer.transform(X_interval))

        print("\n" + "-"*50)
        print(f"{'Значение': <20} | {'Прогноз продаж': <15}")
        print("-"*50)
        for x, y in zip(X_interval.flatten(), y_interval):
            print(f"{x: <20.2f} | {y: <15.2f}")
        print("-"*50)

        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(x_data, y_data, color='gray', alpha=0.5, s=30, label='Реальные данные', zorder=1)
            
            ax.plot(X_interval.flatten(), y_interval, 'r-', linewidth=2.5, marker='o', markersize=5, label='Прогноз модели', zorder=2)
            
            ax.set_xlabel(chosen_feature, fontsize=11)
            ax.set_ylabel('Прогноз объема продаж (monthly_cctv_sales)', fontsize=11)
            ax.set_title(f'Интервальный прогноз для {chosen_feature}', fontsize=12, pad=15)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.savefig('sales_interval_pairwise.png', dpi=300, bbox_inches='tight')
            print("\n✓ График сохранен в sales_interval_pairwise.png")
            plt.show()

def forecast_menu(samples, calculator):
    forecast = ForecastModule()
    if not forecast.load_cctv_data('cctv.csv'):
        return

    print("\n" + "= "*60)
    print("МОДУЛЬ ПРОГНОЗИРОВАНИЯ ОБЪЕМА ПРОДАЖ")
    print("= "*60)

    while True:
        print("\nМЕНЮ ПРОГНОЗИРОВАНИЯ:")
        print("1. Проанализировать данные и выбрать наилучшую модель регрессии")
        print("2. Получить прогноз при ручном вводе ВСЕХ признаков")
        print("3. Интервальный прогноз + График")
        print("0. Вернуться в главное меню")
        
        choice = input("\nВыберите пункт (0-3): ").strip()
        
        if choice == '1':
            forecast.analyze_and_choose_model()
        elif choice == '2':
            forecast.predict_manual()
        elif choice == '3':
            forecast.predict_interval()
        elif choice == '0':
            break
        else:
            print("Неверный выбор")

class OptimizationModule:
    def __init__(self):
        self.c = None
        self.A_ub = None
        self.b_ub = None
        self.A_eq = None
        self.b_eq = None
        self.bounds = [(0, None) for _ in range(6)]
        self.methods = ['highs', 'interior-point', 'revised simplex']
    
    def input_parameters(self):
        print("\n" + "="*60)
        print("ВВОД ПАРАМЕТРОВ ЗАДАЧИ ОПТИМИЗАЦИИ")
        print("="*60)
        
        try:
            print("\nМощности производственных линий:")
            c1 = float(input("  Мощность линии C1 (ед./мес.): "))
            c2 = float(input("  Мощность линии C2 (ед./мес.): "))
            
            print("\nСпрос охранных компаний:")
            sec1 = float(input("  Спрос Sec1 (ед./мес.): "))
            sec2 = float(input("  Спрос Sec2 (ед./мес.): "))
            sec3 = float(input("  Спрос Sec3 (ед./мес.): "))
            
            print("\nСебестоимость и логистика (руб./ед.):")
            print("  Для линии C1:")
            c1_sec1 = float(input("    C1 -> Sec1: "))
            c1_sec2 = float(input("    C1 -> Sec2: "))
            c1_sec3 = float(input("    C1 -> Sec3: "))
            print("  Для линии C2:")
            c2_sec1 = float(input("    C2 -> Sec1: "))
            c2_sec2 = float(input("    C2 -> Sec2: "))
            c2_sec3 = float(input("    C2 -> Sec3: "))
            
            total_capacity = c1 + c2
            total_demand = sec1 + sec2 + sec3
            
            if total_capacity < total_demand:
                print(f"\n ВНИМАНИЕ: Задача НЕДОПУСТИМА!")
                print(f"   Общая мощность: {total_capacity} ед./мес.")
                print(f"   Общий спрос: {total_demand} ед./мес.")
                print(f"   Спрос превышает мощность на {total_demand - total_capacity} ед./мес.")
                print("\nДля решения задачи необходимо:")
                print("   - Увеличить мощности производственных линий, ИЛИ")
                print("   - Уменьшить спрос компаний")
                return False
            
            if total_capacity > total_demand:
                print(f"\n  ИНФОРМАЦИЯ: Избыток мощности")
                print(f"   Общая мощность: {total_capacity} ед./мес.")
                print(f"   Общий спрос: {total_demand} ед./мес.")
                print(f"   Неиспользуемая мощность: {total_capacity - total_demand} ед./мес.")
            
            self.c = [c1_sec1, c1_sec2, c1_sec3, c2_sec1, c2_sec2, c2_sec3]
            self.A_ub = [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1]
            ]
            self.b_ub = [c1, c2]
            self.A_eq = [
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1]
            ]
            self.b_eq = [sec1, sec2, sec3]
            
            print("\n" + "="*60)
            print("ПАРАМЕТРЫ ПРИНЯТЫ")
            print("="*60)
            print(f"Мощности: C1={c1}, C2={c2}")
            print(f"Спрос: Sec1={sec1}, Sec2={sec2}, Sec3={sec3}")
            print(f"Целевая функция: Z = {self.c[0]}x1 + {self.c[1]}x2 + {self.c[2]}x3 + {self.c[3]}x4 + {self.c[4]}x5 + {self.c[5]}x6")
            return True
            
        except ValueError:
            print("Ошибка: введите корректные числовые значения.")
            return False
    
    def solve_optimization(self, method='highs'):
        if self.c is None:
            print("Сначала введите параметры задачи")
            return
            
        print(f"\n>>> Запуск метода: {method}")
        
        start_time = time.time()
        
        res = linprog(c=self.c,
                     A_ub=self.A_ub, b_ub=self.b_ub,
                     A_eq=self.A_eq, b_eq=self.b_eq,
                     bounds=self.bounds,
                     method=method)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if res.success:
            print(f"Статус решения: УСПЕХ ({res.message})")
            print(f"Оптимальный план (x1...x6): {res.x}")
            print(f"Минимальные затраты: {res.fun:.2f} руб.")
            print(f"Время выполнения: {execution_time:.6f} сек.")
        else:
            print(f"Статус решения: ОШИБКА ({res.message})")
        
        return res
    
    def run_all_methods(self):
        if self.c is None:
            print("Сначала введите параметры задачи")
            return
            
        for method in self.methods:
            self.solve_optimization(method)
            print()

def optimization_menu():
    optimizer = OptimizationModule()
    
    print("\n" + "="*60)
    print("               МОДУЛЬ ОПТИМИЗАЦИИ ПРОЦЕССОВ")
    print("="*60)

    print("\nЗАДАЧА: Минимизация затрат на производство и доставку")
    print("камер видеонаблюдения между производственными линиями")
    print("и охранными компаниями")
    
    while True:
        print("\nМЕНЮ ОПТИМИЗАЦИИ:")
        print("1. Ввести параметры задачи")
        print("2. Решить задачу всеми методами")
        print("3. Решить методом HiGHS (рекомендуется)")
        print("4. Решить методом Interior-Point")
        print("5. Решить методом Revised Simplex")
        print("0. Вернуться в главное меню")
        
        choice = input("\nВыберите пункт (0-5): ").strip()
        
        if choice == '1':
            optimizer.input_parameters()
        elif choice == '2':
            optimizer.run_all_methods()
        elif choice == '3':
            optimizer.solve_optimization('highs')
        elif choice == '4':
            optimizer.solve_optimization('interior-point')
        elif choice == '5':
            optimizer.solve_optimization('revised simplex')
        elif choice == '0':
            break
        else:
            print("Неверный выбор")

def main_menu():
    print("\n" + "="*70)
    print("Производитель систем видеонаблюдения")
    print("="*70)
    
    samples = []
    calculator = TechnicalLevelCalculator()
    data_manager = DataManager()
    visualizer = Visualizer()
    
    while True:
        print("\n" + "-"*70)
        print("ГЛАВНОЕ МЕНЮ:")
        print("1. Добавить образец вручную")
        print("2. Загрузить образцы из файла")
        print("3. Показать все образцы")
        print("4. Рассчитать технический уровень")
        print("5. Показать радиальную диаграмму")
        print("6. Показать столбчатую диаграмму")
        print("7. Сохранить данные")
        print("8. Прогнозирование динамики показателей")
        print("9. Оптимизация процессов")
        print("0. Выход")
        print("-"*70)
        
        choice = input("Выберите пункт меню (0-9): ").strip()
        
        if choice == '1':
            samples = add_sample(samples, calculator)
        elif choice == '2':
            samples = load_sample_from_file(samples, calculator)
        elif choice == '3':
            show_all_samples(samples)
        elif choice == '4':
            if samples:
                print("\nПересчет технического уровня...")
                for sample in samples:
                    calculator.calculate_technical_level(sample)
                print(" Расчет завершен!")
                show_all_samples(samples)
            else:
                print("Нет образцов для расчета")
        elif choice == '5':
            if samples:
                visualizer.plot_radial_chart(samples, calculator)
            else:
                print("Нет образцов для отображения")
        elif choice == '6':
            if samples:
                visualizer.plot_bar_chart(samples)
            else:
                print("Нет образцов для отображения")
        elif choice == '7':
            if samples:
                filename = input("Введите имя файла для сохранения (например, data.json или data.csv): ").strip()
                if filename.lower().endswith('.json'):
                    data_manager.save_to_json(samples, filename)
                elif filename.lower().endswith('.csv'):
                    data_manager.save_to_csv(samples, filename)
                else:
                    print("Ошибка: Некорректное расширение. Используйте .json или .csv.")
            else:
                print("Нет данных для сохранения")
        elif choice == '8':
            forecast_menu(samples, calculator)
        elif choice == '9':
            optimization_menu()
        elif choice == '0':
            print("\nСпасибо за работу!")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main_menu()