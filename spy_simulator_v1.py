# =============================================================================
# spy_simulator_v1.py
# SPY ETF 월적립 복리 시뮬레이터
# =============================================================================

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 한글 폰트 설정 (macOS)
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'AppleGothic'  # macOS 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False   # 마이너스(-) 기호 깨짐 방지


# -----------------------------------------------------------------------------
# 2. SPY 역사적 수익률 정보
# -----------------------------------------------------------------------------
# SPY (S&P 500 ETF) 역사적 연평균 수익률
# - 1993년 출시 이후 평균: 약 10.5%
# - 최근 10년 평균: 약 12%
# - 보수적 추정치: 7~8%
# - 일반적 기대치: 10%
SPY_HISTORICAL_RETURN = 0.10  # 연 10% (역사적 평균 기반)


# -----------------------------------------------------------------------------
# 3. 시뮬레이션 함수
# -----------------------------------------------------------------------------
def simulate_investment(monthly, years, annual_return):
    """
    SPY 월적립 복리 시뮬레이터

    Args:
        monthly (int): 월 납입금 (원)
        years (int): 투자 기간 (년)
        annual_return (float): 연 수익률 (예: 0.10 = 10%)

    Returns:
        tuple: (최종 잔고, 연도별 기록 리스트)
    """
    total = 0  # 누적 잔고
    monthly_return = annual_return / 12  # 월 수익률로 환산
    history = []  # 연도별 기록 저장

    # 월별 시뮬레이션 (years * 12개월)
    for month in range(years * 12):
        # Step 1: 월 납입금 추가
        total += monthly

        # Step 2: 이번 달 수익률 적용 (복리)
        total *= (1 + monthly_return)

        # Step 3: 매년 말(12월)에 기록 저장
        if month % 12 == 11:
            year = month // 12 + 1
            history.append({
                'year': year,
                'balance': total
            })

    return total, history


# -----------------------------------------------------------------------------
# 4. 연도별 잔고 출력 함수
# -----------------------------------------------------------------------------
def print_yearly_balance(history, monthly):
    """연도별 잔고를 테이블 형태로 출력"""
    total_invested = 0  # 총 납입 원금

    print("\n" + "=" * 60)
    print("연도별 잔고 현황")
    print("=" * 60)
    print(f"{'연차':<6} {'납입 원금':>15} {'잔고':>18} {'수익률':>10}")
    print("-" * 60)

    for record in history:
        year = record['year']
        balance = record['balance']
        total_invested = monthly * 12 * year  # 해당 연도까지 납입 원금
        profit_rate = ((balance - total_invested) / total_invested) * 100

        print(f"{year:>3}년  {total_invested:>15,.0f}원  {balance:>15,.0f}원  {profit_rate:>8.1f}%")

    print("=" * 60)


# -----------------------------------------------------------------------------
# 5. 그래프 시각화 함수
# -----------------------------------------------------------------------------
def plot_investment_growth(history, monthly, years):
    """투자 성장 그래프 시각화"""
    years_list = [h['year'] for h in history]
    balances = [h['balance'] for h in history]

    # 원금 라인 (비교용)
    principal = [monthly * 12 * y for y in years_list]

    # 그래프 설정
    plt.figure(figsize=(10, 6))

    # 잔고 곡선
    plt.plot(years_list, balances, 'b-', linewidth=2, label='복리 잔고', marker='o')

    # 원금 라인
    plt.plot(years_list, principal, 'r--', linewidth=1.5, label='납입 원금', alpha=0.7)

    # 영역 채우기 (수익 구간)
    plt.fill_between(years_list, principal, balances, alpha=0.3, color='green', label='수익')

    # 라벨 및 제목
    plt.xlabel('투자 기간 (년)', fontsize=12)
    plt.ylabel('금액 (원)', fontsize=12)
    plt.title(f'SPY {years}년 월적립 시뮬레이션 (월 {monthly:,.0f}원)', fontsize=14)

    # 범례 및 그리드
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Y축 포맷 (억 단위 표시)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/100000000:.1f}억' if x >= 100000000 else f'{x/10000:.0f}만'))

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. 메인 실행
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 설정값 (여기서 수정 가능)
    # -------------------------------------------------------------------------
    monthly = 1000000          # 월 납입금: 100만원
    years = 30                 # 투자 기간: 30년
    annual_return = SPY_HISTORICAL_RETURN  # SPY 역사적 수익률 10%

    # -------------------------------------------------------------------------
    # 시뮬레이션 실행
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SPY 월적립 복리 시뮬레이터")
    print("=" * 60)
    print(f"월 납입금: {monthly:,}원")
    print(f"투자 기간: {years}년")
    print(f"예상 연 수익률: {annual_return * 100}% (SPY 역사적 평균)")
    print(f"총 납입 원금: {monthly * 12 * years:,}원")

    # 시뮬레이션 실행
    final_balance, history = simulate_investment(monthly, years, annual_return)

    # -------------------------------------------------------------------------
    # 결과 출력
    # -------------------------------------------------------------------------
    total_principal = monthly * 12 * years
    total_profit = final_balance - total_principal
    profit_rate = (total_profit / total_principal) * 100

    print("\n[ 최종 결과 ]")
    print(f"최종 잔고: {final_balance:,.0f}원")
    print(f"총 수익: {total_profit:,.0f}원")
    print(f"수익률: {profit_rate:.1f}%")

    # 연도별 잔고 출력
    print_yearly_balance(history, monthly)

    # 그래프 시각화
    plot_investment_growth(history, monthly, years)
