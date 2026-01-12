# =============================================================================
# spy_simulator_v2.py
# SPY ETF 월적립 복리 시뮬레이터 (개선 버전)
# =============================================================================
# v2 개선사항:
# - 월 수익률 복리 환산 방식 수정 (연환산 일치)
# - 입력값 검증 강화 (타입, 범위, inflation_rate 포함)
# - 크로스 플랫폼 한글 폰트 지원 (지연 로딩)
# - 그래프 수익/손실 영역 분리 표시
# - Decimal 사용으로 금융 계산 정밀도 향상
# - 인플레이션 조정 실질 수익률 옵션
# - 수치 안정성 향상 (math.log1p/expm1 사용)
# =============================================================================

import sys
import math
import logging
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from typing import List, Optional

# 로깅 설정
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. 크로스 플랫폼 한글 폰트 설정 (지연 로딩)
# -----------------------------------------------------------------------------
_font_initialized = False


def setup_korean_font():
    """OS에 따라 적절한 한글 폰트 설정 (최초 1회만 실행)"""
    global _font_initialized
    if _font_initialized:
        return

    import matplotlib.pyplot as plt

    if sys.platform == "darwin":  # macOS
        candidates = ["AppleGothic", "Apple SD Gothic Neo"]
    elif sys.platform.startswith("win"):  # Windows
        candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic"]
    else:  # Linux
        candidates = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]

    plt.rcParams["font.family"] = candidates
    plt.rcParams["axes.unicode_minus"] = False
    _font_initialized = True


# -----------------------------------------------------------------------------
# 2. 상수 및 데이터 클래스
# -----------------------------------------------------------------------------
# SPY (S&P 500 ETF) 역사적 연평균 수익률
# - 1993년 출시 이후 평균: 약 10.5%
# - 최근 10년 평균: 약 12%
# - 보수적 추정치: 7~8%
SPY_HISTORICAL_RETURN = 0.10  # 연 10%
DEFAULT_INFLATION_RATE = 0.025  # 연 2.5% (한국 평균 인플레이션)


@dataclass
class YearlyRecord:
    """연도별 투자 기록"""
    year: int
    balance: float
    principal: float  # 해당 연도까지 납입 원금
    profit: float
    profit_rate: float  # 퍼센트
    real_balance: Optional[float] = None  # 인플레이션 조정 실질 가치


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    final_balance: float
    total_principal: float
    total_profit: float
    profit_rate: float
    history: List[YearlyRecord]
    monthly_return_used: float  # 실제 사용된 월 수익률
    real_final_balance: Optional[float] = None  # 인플레이션 조정 값


# -----------------------------------------------------------------------------
# 3. 유틸리티 함수
# -----------------------------------------------------------------------------
def annual_to_monthly_return(annual_return: float) -> float:
    """
    연 수익률을 월 수익률로 복리 환산 (수치 안정성 향상)

    복리 공식: (1 + 연수익률) = (1 + 월수익률)^12
    따라서: 월수익률 = (1 + 연수익률)^(1/12) - 1

    수치 안정성을 위해 log1p/expm1 사용:
    월수익률 = expm1(log1p(연수익률) / 12)
    """
    return math.expm1(math.log1p(annual_return) / 12.0)


def validate_inputs(
    monthly: int,
    years: int,
    annual_return: float,
    inflation_rate: Optional[float] = None
) -> None:
    """입력값 검증 (타입 및 범위)"""
    # years 검증
    if not isinstance(years, int) or years <= 0:
        raise ValueError(f"투자 기간(years)은 1 이상의 정수여야 합니다. 입력값: {years}")

    # monthly 타입 및 범위 검증
    if not isinstance(monthly, (int, float)):
        raise TypeError(f"월 납입금(monthly)은 숫자여야 합니다. 입력 타입: {type(monthly).__name__}")
    if monthly < 0:
        raise ValueError(f"월 납입금(monthly)은 0 이상이어야 합니다. 입력값: {monthly}")

    # annual_return 타입 및 범위 검증
    if not isinstance(annual_return, (int, float)):
        raise TypeError(f"연 수익률(annual_return)은 숫자여야 합니다. 입력 타입: {type(annual_return).__name__}")
    if annual_return <= -1.0:
        raise ValueError(f"연 수익률(annual_return)은 -100%보다 커야 합니다. 입력값: {annual_return}")
    if annual_return > 1.0:  # 100% 초과 경고
        logger.warning(f"연 수익률 {annual_return * 100}%는 비현실적으로 높습니다.")

    # inflation_rate 검증 (None이 아닌 경우)
    if inflation_rate is not None:
        if not isinstance(inflation_rate, (int, float)):
            raise TypeError(f"인플레이션율(inflation_rate)은 숫자여야 합니다. 입력 타입: {type(inflation_rate).__name__}")
        if inflation_rate <= -1.0:
            raise ValueError(f"인플레이션율(inflation_rate)은 -100%보다 커야 합니다. 입력값: {inflation_rate}")


# -----------------------------------------------------------------------------
# 4. 시뮬레이션 함수
# -----------------------------------------------------------------------------
def simulate_investment(
    monthly: int,
    years: int,
    annual_return: float,
    inflation_rate: Optional[float] = None
) -> SimulationResult:
    """
    SPY 월적립 복리 시뮬레이터 (개선 버전)

    Args:
        monthly: 월 납입금 (원)
        years: 투자 기간 (년)
        annual_return: 연 수익률 (예: 0.10 = 10%)
        inflation_rate: 연 인플레이션율 (None이면 계산 안 함)

    Returns:
        SimulationResult: 시뮬레이션 결과 객체
    """
    # 입력값 검증 (inflation_rate 포함)
    validate_inputs(monthly, years, annual_return, inflation_rate)

    # 복리 기준 월 수익률 환산 (핵심 수정!)
    monthly_return = annual_to_monthly_return(annual_return)

    # 인플레이션 월 환산
    monthly_inflation = None
    if inflation_rate is not None:
        monthly_inflation = annual_to_monthly_return(inflation_rate)

    total = Decimal("0")  # 정밀도를 위해 Decimal 사용
    monthly_dec = Decimal(str(monthly))
    monthly_return_dec = Decimal(str(monthly_return))

    history: List[YearlyRecord] = []

    # 인플레이션 누적 계수 (현재 가치 → 미래 가치)
    inflation_factor = Decimal("1")
    if monthly_inflation is not None:
        monthly_inflation_dec = Decimal(str(monthly_inflation))

    # 월별 시뮬레이션
    for month in range(years * 12):
        # Step 1: 월 납입금 추가
        total += monthly_dec

        # Step 2: 이번 달 수익률 적용 (복리)
        total *= (Decimal("1") + monthly_return_dec)

        # Step 3: 인플레이션 누적
        if monthly_inflation is not None:
            inflation_factor *= (Decimal("1") + monthly_inflation_dec)

        # Step 4: 매년 말(12월)에 기록 저장
        if month % 12 == 11:
            year = month // 12 + 1
            balance = float(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            principal = monthly * 12 * year
            profit = balance - principal
            profit_rate = (profit / principal * 100) if principal > 0 else 0.0

            # 실질 가치 (인플레이션 조정)
            real_balance = None
            if monthly_inflation is not None:
                real_balance = float(
                    (total / inflation_factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                )

            history.append(YearlyRecord(
                year=year,
                balance=balance,
                principal=principal,
                profit=profit,
                profit_rate=profit_rate,
                real_balance=real_balance
            ))

    # 최종 결과
    final_balance = float(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    total_principal = monthly * 12 * years
    total_profit = final_balance - total_principal
    final_profit_rate = (total_profit / total_principal * 100) if total_principal > 0 else 0.0

    real_final_balance = None
    if monthly_inflation is not None:
        real_final_balance = float(
            (total / inflation_factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )

    return SimulationResult(
        final_balance=final_balance,
        total_principal=total_principal,
        total_profit=total_profit,
        profit_rate=final_profit_rate,
        history=history,
        monthly_return_used=monthly_return,
        real_final_balance=real_final_balance
    )


# -----------------------------------------------------------------------------
# 5. 출력 함수
# -----------------------------------------------------------------------------
def print_simulation_config(monthly: int, years: int, annual_return: float,
                            inflation_rate: Optional[float] = None):
    """시뮬레이션 설정 출력"""
    print("\n" + "=" * 70)
    print("SPY 월적립 복리 시뮬레이터 v2")
    print("=" * 70)
    print(f"월 납입금: {monthly:,}원")
    print(f"투자 기간: {years}년 ({years * 12}개월)")
    print(f"예상 연 수익률: {annual_return * 100:.1f}% (SPY 역사적 평균)")
    print(f"월 복리 수익률: {annual_to_monthly_return(annual_return) * 100:.4f}%")
    print(f"총 납입 원금: {monthly * 12 * years:,}원")
    if inflation_rate is not None:
        print(f"예상 인플레이션: {inflation_rate * 100:.1f}%/년")


def print_result_summary(result: SimulationResult):
    """최종 결과 요약 출력"""
    print("\n" + "=" * 70)
    print("[ 최종 결과 ]")
    print("=" * 70)
    print(f"최종 잔고: {result.final_balance:,.0f}원")
    print(f"총 납입 원금: {result.total_principal:,}원")
    print(f"총 수익: {result.total_profit:,.0f}원")
    print(f"수익률: {result.profit_rate:.1f}%")

    if result.real_final_balance is not None:
        print(f"\n[ 인플레이션 조정 실질 가치 ]")
        print(f"실질 최종 잔고: {result.real_final_balance:,.0f}원 (현재 화폐 가치 기준)")
        real_profit = result.real_final_balance - result.total_principal
        real_rate = (real_profit / result.total_principal * 100) if result.total_principal > 0 else 0
        print(f"실질 수익률: {real_rate:.1f}%")


def print_yearly_balance(result: SimulationResult, show_real: bool = False):
    """연도별 잔고를 테이블 형태로 출력"""
    print("\n" + "=" * 70)
    print("연도별 잔고 현황")
    print("=" * 70)

    if show_real and result.real_final_balance is not None:
        print(f"{'연차':<5} {'납입 원금':>14} {'명목 잔고':>16} {'실질 잔고':>16} {'수익률':>8}")
    else:
        print(f"{'연차':<5} {'납입 원금':>14} {'잔고':>18} {'수익률':>10}")
    print("-" * 70)

    for record in result.history:
        if show_real and record.real_balance is not None:
            print(f"{record.year:>3}년  {record.principal:>14,.0f}원  "
                  f"{record.balance:>14,.0f}원  {record.real_balance:>14,.0f}원  "
                  f"{record.profit_rate:>6.1f}%")
        else:
            print(f"{record.year:>3}년  {record.principal:>14,.0f}원  "
                  f"{record.balance:>16,.0f}원  {record.profit_rate:>8.1f}%")

    print("=" * 70)


# -----------------------------------------------------------------------------
# 6. 그래프 시각화 함수
# -----------------------------------------------------------------------------
def plot_investment_growth(result: SimulationResult, monthly: int, years: int,
                           show_real: bool = False):
    """투자 성장 그래프 시각화 (개선 버전)"""
    import matplotlib.pyplot as plt
    setup_korean_font()

    years_list = [r.year for r in result.history]
    balances = [r.balance for r in result.history]
    principals = [r.principal for r in result.history]

    # 그래프 설정
    fig, ax = plt.subplots(figsize=(12, 7))

    # 잔고 곡선
    ax.plot(years_list, balances, 'b-', linewidth=2.5, label='명목 잔고', marker='o', markersize=4)

    # 실질 잔고 (인플레이션 조정)
    if show_real and result.history[0].real_balance is not None:
        real_balances = [r.real_balance for r in result.history]
        ax.plot(years_list, real_balances, 'g-', linewidth=2, label='실질 잔고 (인플레이션 조정)',
                marker='s', markersize=4, alpha=0.8)

    # 원금 라인
    ax.plot(years_list, principals, 'r--', linewidth=2, label='납입 원금', alpha=0.7)

    # 수익 구간 채우기 (잔고 >= 원금인 경우만)
    ax.fill_between(
        years_list, principals, balances,
        where=[b >= p for b, p in zip(balances, principals)],
        interpolate=True,
        alpha=0.25, color='green', label='수익 구간'
    )

    # 손실 구간 채우기 (잔고 < 원금인 경우)
    ax.fill_between(
        years_list, principals, balances,
        where=[b < p for b, p in zip(balances, principals)],
        interpolate=True,
        alpha=0.25, color='red', label='손실 구간'
    )

    # 주요 마일스톤 표시 (5년 단위)
    for record in result.history:
        if record.year % 5 == 0 or record.year == years:
            ax.annotate(
                f'{record.balance/100000000:.1f}억',
                xy=(record.year, record.balance),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='blue'
            )

    # 라벨 및 제목
    ax.set_xlabel('투자 기간 (년)', fontsize=12)
    ax.set_ylabel('금액 (원)', fontsize=12)
    ax.set_title(f'SPY {years}년 월적립 시뮬레이션 (월 {monthly:,.0f}원, 연 {result.profit_rate/years:.1f}% 평균)',
                 fontsize=14, fontweight='bold')

    # 범례 및 그리드
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Y축 포맷 (억/만 단위 자동 표시)
    def format_currency(x, p):
        if x >= 100000000:
            return f'{x/100000000:.1f}억'
        elif x >= 10000:
            return f'{x/10000:.0f}만'
        else:
            return f'{x:.0f}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_currency))

    # X축 설정
    ax.set_xlim(0, years + 1)
    ax.set_xticks(range(0, years + 1, 5 if years > 10 else 1))

    plt.tight_layout()
    plt.show()


def plot_comparison(scenarios: List[tuple], monthly: int):
    """여러 시나리오 비교 그래프"""
    import matplotlib.pyplot as plt
    setup_korean_font()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    for i, (name, result) in enumerate(scenarios):
        years_list = [r.year for r in result.history]
        balances = [r.balance for r in result.history]
        color = colors[i % len(colors)]
        ax.plot(years_list, balances, linewidth=2, label=name, marker='o', markersize=3, color=color)

    # 원금 라인 (첫 번째 시나리오 기준)
    if scenarios:
        principals = [r.principal for r in scenarios[0][1].history]
        ax.plot(years_list, principals, 'r--', linewidth=2, label='납입 원금', alpha=0.5)

    ax.set_xlabel('투자 기간 (년)', fontsize=12)
    ax.set_ylabel('금액 (원)', fontsize=12)
    ax.set_title(f'수익률 시나리오 비교 (월 {monthly:,.0f}원 적립)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/100000000:.1f}억' if x >= 100000000 else f'{x/10000:.0f}만'
    ))

    plt.tight_layout()
    plt.show()


# =============================================================================
# 7. 메인 실행
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 설정값
    # -------------------------------------------------------------------------
    monthly = 1_000_000              # 월 납입금: 100만원
    years = 30                       # 투자 기간: 30년
    annual_return = SPY_HISTORICAL_RETURN  # SPY 역사적 수익률 10%
    inflation_rate = DEFAULT_INFLATION_RATE  # 인플레이션 2.5%

    # -------------------------------------------------------------------------
    # 시뮬레이션 실행
    # -------------------------------------------------------------------------
    print_simulation_config(monthly, years, annual_return, inflation_rate)

    result = simulate_investment(monthly, years, annual_return, inflation_rate)

    # 결과 출력
    print_result_summary(result)
    print_yearly_balance(result, show_real=True)

    # 그래프 시각화
    plot_investment_growth(result, monthly, years, show_real=True)

    # -------------------------------------------------------------------------
    # 시나리오 비교 (선택적)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[ 수익률 시나리오 비교 ]")
    print("=" * 70)

    scenarios = [
        ("보수적 (7%)", simulate_investment(monthly, years, 0.07)),
        ("평균 (10%)", simulate_investment(monthly, years, 0.10)),
        ("낙관적 (12%)", simulate_investment(monthly, years, 0.12)),
    ]

    for name, res in scenarios:
        print(f"{name}: 최종 잔고 {res.final_balance:,.0f}원 (수익률 {res.profit_rate:.1f}%)")

    # 시나리오 비교 그래프
    plot_comparison(scenarios, monthly)
