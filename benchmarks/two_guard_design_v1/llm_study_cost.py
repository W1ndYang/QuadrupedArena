"""Public list-price estimates from reported request usage, in CNY."""
import math


PRICE_REFERENCE = {
    'checked_on': '2026-11-19',
    'currency': 'CNY',
    'deepseek-flash': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
    'qwen3.6-27b': 'https://help.aliyun.com/zh/model-studio/qwen3-6-27b',
    'scope': 'DeepSeek published off-peak to peak range, Qwen mainland list price. '
             'Excludes account discounts, credits and calls without usage.',
}


def public_price(response, config):
    if not response or response.get('response_source') != 'live_api':
        return None
    usage = response.get('usage') or {}
    inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if any(type(v) is not int or v < 0 for v in (inp, out)):
        return None
    model = config.get('model')
    if model == 'qwen3.6-27b' and config.get('base_url', '').startswith(
            'https://dashscope.aliyuncs.com/'):
        price = (3*inp+18*out)/1_000_000
        return [price, price]
    if model == 'deepseek-flash':
        hit, miss = usage.get('prompt_cache_hit_tokens'), usage.get('prompt_cache_miss_tokens')
        if type(hit) is int and type(miss) is int and min(hit, miss) >= 0 and hit+miss == inp:
            lower = (.02*hit+1*miss+4*out)/1_000_000
            return [lower, lower*2]
        return [(.02*inp+4*out)/1_000_000, (2*inp+8*out)/1_000_000]
    return None


def sum_prices(prices):
    reported = [price for price in prices if price is not None]
    return dict(calls_with_reported_cost_estimate=len(reported),
                cny_interval=[math.fsum(price[index] for price in reported) for index in (0,1)]
                if reported else None)
