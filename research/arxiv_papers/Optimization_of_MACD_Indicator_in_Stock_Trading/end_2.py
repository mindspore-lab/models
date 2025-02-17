import os
import numpy as np
import pandas as pd
import pywt


def find_local_extrema(data, window, greater=True):
    special_indices = []
    for i in range(window + 1, len(data)):
        window_data = data[i - window - 1:i]
        if greater:
            if data[i - 1] == window_data.max() and data[i] < data[i - 1]:
                special_indices.append(i - 1)
        else:
            if data[i - 1] == window_data.min() and data[i] > data[i - 1]:
                special_indices.append(i - 1)

    return np.array(special_indices)


def genetic_algorithm(fitness_func, bounds, num_individuals, num_generations, crossover_rate, mutation_rate,
                      max_stagnant_generations, subset_data1):
    num_variables = len(bounds)
    population = np.random.randint(low=[b[0] for b in bounds], high=[b[1] + 1 for b in bounds],
                                   size=(num_individuals, num_variables))
    elite_size = max(1, num_individuals // 10)

    def evaluate_population(pop):
        return np.array([fitness_func(subset_data1, ind) for ind in pop])

    def select(population, fitness):

        elite_indices = np.argsort(fitness)[-elite_size:]
        elite = population[elite_indices]

        max_fitness = np.max(fitness)
        exp_values = np.exp(fitness - max_fitness)
        probabilities = exp_values / exp_values.sum()

        chosen_indices = np.random.choice(num_individuals, size=num_individuals - elite_size, replace=True,
                                          p=probabilities)
        return np.concatenate((population[chosen_indices], elite))

    best_fitness_history = []
    stagnant_counter = 0
    for generation in range(num_generations):
        fitness = evaluate_population(population)
        best_index = np.argmax(fitness)
        max_fitness = fitness.max()
        best_fitness_history.append(max_fitness)
        print(population[best_index])

        if max_stagnant_generations:
            if len(best_fitness_history) > 1 and (best_fitness_history[-1] - best_fitness_history[-2] < 1e-5):
                stagnant_counter += 1
            else:
                stagnant_counter = 0
            if stagnant_counter >= max_stagnant_generations:
                print(f"Progress stagnant for {max_stagnant_generations} generations at generation {generation}")
                break

        selected = select(population, fitness)
        new_population = selected.copy()

        for i in range(0, num_individuals - elite_size, 2):
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, num_variables)
                new_population[i, point:], new_population[i + 1, point:] = new_population[i + 1,
                                                                           point:], new_population[i, point:].copy()
            new_population[i] = mutate(new_population[i], mutation_rate, bounds)
            new_population[i + 1] = mutate(new_population[i + 1], mutation_rate, bounds)

        population = new_population
        print(f"Generation {generation}: Max Fitness = {fitness.max()}")

    fitness = evaluate_population(population)
    best_index = np.argmax(fitness)
    return population[best_index], fitness.max()


def mutate(individual, mutation_rate, bounds):
    for i in range(len(bounds)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(bounds[i][0], bounds[i][1] + 1)
    return individual


def fitness_function(data, individual):
    x, y, z = individual
    ema_short = data['收盘价(元)'].ewm(span=x, adjust=False).mean()
    ema_long = data['收盘价(元)'].ewm(span=y, adjust=False).mean()
    dif = ema_short - ema_long
    dea = dif.ewm(span=z, adjust=False).mean()
    macd_histogram = dif - dea
    coeffs = pywt.wavedec(dif.values, 'coif5', level=4)

    approximation = coeffs[0]

    reconstructed_signal = pywt.waverec([approximation] + [np.zeros_like(coeff) for coeff in coeffs[1:]], 'coif5')
    if len(reconstructed_signal) != len(dif):
        reconstructed_signal = reconstructed_signal[:len(dif)]
    reconstructed_dif = pd.Series(reconstructed_signal, index=dif.index)

    data['DIF'] = reconstructed_dif
    data['DEA'] = dea
    data['MACD'] = macd_histogram

    buy_signals = (data['DIF'] > data['MACD']) & (data['DIF'].shift(1) <= data['MACD'].shift(1))
    sell_signals = (data['DIF'] < data['MACD']) & (data['DIF'].shift(1) >= data['MACD'].shift(1))

    order = 15

    price_max_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=True)
    price_min_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=False)
    macd_max_idx = find_local_extrema(data['MACD'], window=order, greater=True)
    macd_min_idx = find_local_extrema(data['MACD'], window=order, greater=False)

    data['price_max'] = pd.Series(data['收盘价(元)'].iloc[price_max_idx].values, index=price_max_idx)
    data['price_min'] = pd.Series(data['收盘价(元)'].iloc[price_min_idx].values, index=price_min_idx)
    data['macd_max'] = pd.Series(data['MACD'].iloc[macd_max_idx].values, index=macd_max_idx)
    data['macd_min'] = pd.Series(data['MACD'].iloc[macd_min_idx].values, index=macd_min_idx)

    data['bull_divergence'] = False
    data['bear_divergence'] = False

    for i in range(1, len(price_max_idx)):
        current_idx = price_max_idx[i]
        previous_idx = price_max_idx[i - 1]

        if data['收盘价(元)'].iloc[current_idx] > data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
            current_idx] < data['MACD'].iloc[previous_idx]:
            data.loc[current_idx, 'bull_divergence'] = True

    for i in range(1, len(macd_max_idx)):
        current_idx = macd_max_idx[i]
        previous_idx = macd_max_idx[i - 1]

        if data['收盘价(元)'].iloc[current_idx] > data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
            current_idx] < data['MACD'].iloc[previous_idx]:
            data.loc[current_idx, 'bull_divergence'] = True

    for i in range(1, len(price_min_idx)):
        current_idx = price_min_idx[i]
        previous_idx = price_min_idx[i - 1]

        if data['收盘价(元)'].iloc[current_idx] < data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
            current_idx] > data['MACD'].iloc[previous_idx]:
            data.loc[current_idx, 'bear_divergence'] = True

    for i in range(1, len(macd_min_idx)):
        current_idx = macd_min_idx[i]
        previous_idx = macd_min_idx[i - 1]

    if data['收盘价(元)'].iloc[current_idx] < data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[current_idx] > \
            data['MACD'].iloc[previous_idx]:
        data.loc[current_idx, 'bear_divergence'] = True

    initial_capital = 500000.0
    capital = initial_capital
    stocks_held = 0

    buy_price = 0
    for i in range(len(data) - 1):
        if (buy_signals.iloc[i] or data['bear_divergence'][i]) and capital > 0:
            num_shares_to_buy = capital // data['收盘价(元)'].iloc[i]
            buy_price += num_shares_to_buy * data['收盘价(元)'].iloc[i]
            capital -= num_shares_to_buy * data['收盘价(元)'].iloc[i]
            stocks_held += num_shares_to_buy

        elif (sell_signals.iloc[i] or data['bull_divergence'][i]) and stocks_held > 0:
            num_shares_to_sell = stocks_held
            sell_price = num_shares_to_sell * data['收盘价(元)'].iloc[i]
            capital += sell_price
            stocks_held = 0
            buy_price = 0

    if stocks_held > 0:
        num_shares_to_sell = stocks_held
        capital += num_shares_to_sell * data['收盘价(元)'].iloc[len(data) - 1]
        stocks_held = 0

    total_return = capital - initial_capital

    return total_return


bounds = [(5, 20), (20, 50), (5, 25)]
NUM_INDIVIDUALS = 256
NUM_GENERATIONS = 1024
CROSSOVER_RATE = 0.95
MUTATION_RATE = 0.05

folder_path = 'data'
ans = []
for file in os.listdir(folder_path):
    if file.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file)
        data = pd.read_excel(file_path)
        data = data.iloc[:-2]
        base_name, _ = os.path.splitext(file)

        best_solution, max_fitness = genetic_algorithm(fitness_function, bounds, NUM_INDIVIDUALS, NUM_GENERATIONS,
                                                       CROSSOVER_RATE, MUTATION_RATE, 8, data)
        print("Best Solution:", best_solution)
        print("Max Fitness:", max_fitness)

        x, y, z = best_solution

        ema_short = data['收盘价(元)'].ewm(span=x, adjust=False).mean()
        ema_long = data['收盘价(元)'].ewm(span=y, adjust=False).mean()
        dif = ema_short - ema_long
        dea = dif.ewm(span=z, adjust=False).mean()
        macd_histogram = dif - dea
        coeffs = pywt.wavedec(dif.values, 'coif5', level=4)

        approximation = coeffs[0]

        reconstructed_signal = pywt.waverec([approximation] + [np.zeros_like(coeff) for coeff in coeffs[1:]], 'coif5')
        if len(reconstructed_signal) != len(dif):
            reconstructed_signal = reconstructed_signal[:len(dif)]
        reconstructed_dif = pd.Series(reconstructed_signal, index=dif.index)

        data['DIF'] = reconstructed_dif
        data['DEA'] = dea
        data['MACD'] = macd_histogram

        buy_signals = (data['DIF'] > data['MACD']) & (data['DIF'].shift(1) <= data['MACD'].shift(1))
        sell_signals = (data['DIF'] < data['MACD']) & (data['DIF'].shift(1) >= data['MACD'].shift(1))

        order = 15

        price_max_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=True)
        price_min_idx = find_local_extrema(data['收盘价(元)'], window=order, greater=False)
        macd_max_idx = find_local_extrema(data['MACD'], window=order, greater=True)
        macd_min_idx = find_local_extrema(data['MACD'], window=order, greater=False)

        data['price_max'] = pd.Series(data['收盘价(元)'].iloc[price_max_idx].values, index=price_max_idx)
        data['price_min'] = pd.Series(data['收盘价(元)'].iloc[price_min_idx].values, index=price_min_idx)
        data['macd_max'] = pd.Series(data['MACD'].iloc[macd_max_idx].values, index=macd_max_idx)
        data['macd_min'] = pd.Series(data['MACD'].iloc[macd_min_idx].values, index=macd_min_idx)

        data['bull_divergence'] = False
        data['bear_divergence'] = False

        for i in range(1, len(price_max_idx)):
            current_idx = price_max_idx[i]
            previous_idx = price_max_idx[i - 1]

            if data['收盘价(元)'].iloc[current_idx] > data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
                current_idx] < data['MACD'].iloc[previous_idx]:
                data.loc[current_idx, 'bull_divergence'] = True

        for i in range(1, len(macd_max_idx)):
            current_idx = macd_max_idx[i]
            previous_idx = macd_max_idx[i - 1]

            if data['收盘价(元)'].iloc[current_idx] > data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
                current_idx] < data['MACD'].iloc[previous_idx]:
                data.loc[current_idx, 'bull_divergence'] = True

        for i in range(1, len(price_min_idx)):
            current_idx = price_min_idx[i]
            previous_idx = price_min_idx[i - 1]

            if data['收盘价(元)'].iloc[current_idx] < data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
                current_idx] > data['MACD'].iloc[previous_idx]:
                data.loc[current_idx, 'bear_divergence'] = True

        for i in range(1, len(macd_min_idx)):
            current_idx = macd_min_idx[i]
            previous_idx = macd_min_idx[i - 1]

        if data['收盘价(元)'].iloc[current_idx] < data['收盘价(元)'].iloc[previous_idx] and data['MACD'].iloc[
            current_idx] > data['MACD'].iloc[previous_idx]:
            data.loc[current_idx, 'bear_divergence'] = True

        initial_capital = 500000.0
        capital = initial_capital
        stocks_held = 0
        portfolio_values = []

        trades_tot = 0
        buy_price = 0
        trades = 0
        gains = []
        for i in range(len(data) - 1):
            if (buy_signals.iloc[i] or data['bear_divergence'][i]) and capital > 0:
                num_shares_to_buy = capital // data['收盘价(元)'].iloc[i]
                buy_price += num_shares_to_buy * data['收盘价(元)'].iloc[i]
                capital -= num_shares_to_buy * data['收盘价(元)'].iloc[i]
                stocks_held += num_shares_to_buy
                trades_tot += 1

            elif (sell_signals.iloc[i] or data['bull_divergence'][i]) and stocks_held > 0:
                num_shares_to_sell = stocks_held
                sell_price = num_shares_to_sell * data['收盘价(元)'].iloc[i]
                capital += sell_price
                stocks_held = 0
                trades += 1
                trades_tot += 1
                gains.append(sell_price - buy_price)
                buy_price = 0

            portfolio_value = capital + stocks_held * data['收盘价(元)'][i]
            portfolio_values.append(portfolio_value)

        if stocks_held > 0:
            num_shares_to_sell = stocks_held
            capital += num_shares_to_sell * data['收盘价(元)'].iloc[len(data) - 1]
            stocks_held = 0
            trades += 1
            trades_tot += 1
            gains.append(sell_price - buy_price)

        portfolio_value = capital + stocks_held * data['收盘价(元)'][len(data) - 1]
        portfolio_values.append(portfolio_value)

        gains = np.array(gains)
        profits = gains[gains > 0]
        losses = gains[gains < 0]

        win_rate = len(profits) / trades if trades > 0 else 0
        avg_gain = np.mean(profits) if len(profits) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        odds_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0
        trade_frequency = trades_tot / len(data) if len(data) > 0 else 0

        total_return = capital - initial_capital
        annual_return = ((total_return / initial_capital) + 1) ** (252 / len(data)) - 1
        portfolio_values = np.array(portfolio_values)

        annual_risk_free_rate = 2.653 / 100

        daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        adjusted_returns = returns - daily_risk_free_rate
        sharpe_ratio = (np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(len(data))
                        if np.std(adjusted_returns) != 0 else 0)
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(
            portfolio_values) if np.max(portfolio_values) > 0 else 0

        print({
            "name": base_name,
            "parameter": best_solution,
            "win_rate": win_rate,
            "odds_ratio": odds_ratio,
            "trade_frequency": trade_frequency,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        })
        ans.append({
            "name": base_name,
            "parameter": best_solution,
            "win_rate": win_rate,
            "odds_ratio": odds_ratio,
            "trade_frequency": trade_frequency,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        })
ansdf = pd.DataFrame(ans)

file_path = "x4.xlsx"
ansdf.to_excel(file_path, index=False)