#Начало
import json
import csv
import os
from datetime import datetime

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
        print(f"✓ Диаграмма сохранена: radial_{safe_name}.png")
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
        ax.legend(handles=legend_elements, loc='lower right')
        
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
    print("="*50)
    
    while True:
        choice = input("Введите номер (1-3): ").strip()
        if choice == '1':
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
            value = input(f"  {char_name} [{default_value}]: ").strip()
            if value == '':
                value = default_value
            try:
                value = float(value)
                characteristics[char_name] = value
                break
            except ValueError:
                print("  Ошибка! Введите числовое значение.")
    
    return characteristics

def add_sample(samples, calculator):
    print("\n" + "="*50)
    print("ДОБАВЛЕНИЕ НОВОГО ОБРАЗЦА")
    print("="*50)
    
    name = input("Введите название образца: ").strip()
    if not name:
        name = f"Образец_{len(samples) + 1}"
    
    product_type = get_product_type()
    
    sample = ProductSample(name, product_type)
    
    characteristics = get_characteristics(product_type)
    
    weights = calculator.weights.get(product_type, {})
    for char_name, value in characteristics.items():
        weight = weights.get(char_name, 1.0)
        sample.add_characteristic(char_name, value, weight)
    
    calculator.calculate_technical_level(sample)
    
    samples.append(sample)
    
    print(f"\n✓ Образец '{name}' добавлен!")
    print(f"  Технический уровень: {sample.technical_level:.3f}")
    
    return samples

def load_sample_from_file(samples, calculator):
    print("\n" + "="*50)
    print("ЗАГРУЗКА ИЗ ФАЙЛА")
    print("="*50)
    print("1. Загрузить из JSON")
    print("2. Загрузить из CSV")
    
    choice = input("Выберите формат (1-2): ").strip()
    
    if choice == '1':
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

def main_menu():
    print("\n" + "="*70)
    print("ПРОГРАММА ОЦЕНКИ ТЕХНИЧЕСКОГО УРОВНЯ ПРОДУКЦИИ")
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
        print("0. Выход")
        print("-"*70)
        
        choice = input("Выберите пункт меню (0-7): ").strip()
        
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
                print("✓ Расчет завершен!")
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
                data_manager.save_to_json(samples)
                data_manager.save_to_csv(samples)
            else:
                print("Нет данных для сохранения")
        elif choice == '0':
            print("\nСпасибо за работу!")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main_menu()

