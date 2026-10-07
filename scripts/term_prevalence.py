"""Developer tooling: rough per-node prevalence in the canonical corpus by regex term matching.
Used to decide which v1.0 nodes to merge for v1.1 (spec §0.8). Upper-bound-ish, not labels."""
import polars as pl, json, time
T = {
 "treasury.funding": r"covered bond|senior (unsecured|preferred|non-preferred) (bond|note)s? .{0,40}bank|bank funding|wholesale funding|funding cost|interbank funding",
 "treasury.liquidity": r"liquidity coverage ratio|\blcr\b|high-quality liquid assets|\bhqla\b|liquidity buffer",
 "treasury.structural_rates": r"net interest income|\birrbb\b|interest-rate risk in the banking book|asset-liability management|hedg\w+ .{0,30}(securities|bond) portfolio",
 "treasury.structural_fx": r"structural (fx|foreign[- ]exchange|currency)|translation (risk|exposure)|hedg\w+ .{0,20}capital .{0,20}currenc",
 "markets.macro.rates": r"interest[- ]rate swaps?|swaptions?|primary dealers?|rates (desk|trading)|bond[- ]trading revenue|fixed[- ]income trading",
 "markets.macro.fx": r"(currency|foreign[- ]exchange|\bfx\b) (trading|desk|dealers?|traders? at)|fx (forwards?|swaps?|options?)",
 "markets.macro.commodities": r"commodit\w+ (trading|desk|traders? at|derivatives?)|(oil|gas|metals?) (trading desk|swaps?)",
 "markets.macro.inflation": r"inflation[- ](swaps?|linked)|\btips\b|\blinkers?\b|breakeven",
 "markets.macro.emerging_markets": r"emerging[- ]market (debt|currenc\w+|bonds?) .{0,40}(trading|desk|dealers?|bank)|local[- ]currency (bonds?|debt)",
 "markets.credit.investment_grade": r"investment[- ]grade (bonds?|debt|credit)|high[- ]grade (bonds?|debt)",
 "markets.credit.high_yield_distressed": r"high[- ]yield|junk bonds?|distressed (debt|bonds?)",
 "markets.credit.credit_derivatives": r"credit[- ]default swaps?|\bcds\b|\bcdx\b|itraxx|credit derivatives?",
 "markets.credit.structured_credit": r"synthetic cdo|bespoke tranche|correlation trad\w+|tranche[sd]? .{0,20}credit",
 "markets.equities.cash": r"block (trade|sale)|accelerated bookbuild|equity (trading|execution|desk)|equities trading",
 "markets.equities.derivatives": r"equity derivatives?|equity (options?|swaps?)|index options?|variance swaps?",
 "markets.equities.structured": r"structured (notes?|products?)|equity[- ]linked notes?|autocallable",
 "markets.securitised.rmbs": r"\brmbs\b|residential mortgage[- ]backed|mortgage[- ]backed securit|agency mbs|non-agency",
 "markets.securitised.cmbs": r"\bcmbs\b|commercial mortgage[- ]backed",
 "markets.securitised.abs": r"\babs\b|asset[- ]backed (securit|bonds?|notes?)",
 "markets.securitised.clo_cdo": r"\bclos?\b|collateralized loan obligations?|\bcdos?\b|collateralized debt obligations?",
 "markets.financing.repo": r"\brepo\b|repurchase agreements?|reverse repo",
 "markets.financing.securities_lending": r"securities lending|stock lending|stock[- ]borrow|lend\w* .{0,15}shares to short",
 "markets.financing.prime": r"prime brokerage|prime brokers?",
 "markets.financing.margin": r"margin (loans?|lending|calls?)|securities[- ]based lending",
 "markets.financing.synthetic": r"total[- ]return swaps?|contracts? for difference|synthetic prime",
 "markets.derivatives.otc": r"over[- ]the[- ]counter derivatives?|\botc derivatives?|swaps? dealers?|derivatives? (dealers?|counterpart)",
 "markets.derivatives.cleared": r"(centrally|central) clear\w*|clearinghouses?|clearing houses?|\bccps?\b",
 "markets.derivatives.client_clearing": r"(futures commission merchants?|\bfcms?\b)|client clearing|clearing (members?|services?) for clients",
 "markets.derivatives.xva": r"\bcva\b|\bxva\b|\bfva\b|credit valuation adjustment|valuation adjustment",
 "markets.underwriting.dcm": r"(bond|debt|note) (sale|offering|issue)s? .{0,80}(managed|arranged|led|underwritten) by|bookrunners?|underwriters? (of|on|for) .{0,30}(bonds?|notes?|debt)",
 "markets.underwriting.ecm": r"initial public offering|\bipos?\b|follow[- ]on offering|secondary offering|rights (offer|issue)|convertible (bonds?|notes?)",
 "markets.underwriting.loan_syndication": r"syndicated loans?|loan syndicat\w+|mandated lead arrangers?|\bmlas?\b|syndicat\w+ .{0,20}loan",
 "lending.corporate.large": r"revolving credit (facility|line)|\brevolver\b|credit facility|term loan",
 "lending.corporate.middle_market": r"middle[- ]market|mid[- ]market (compan|lend|loan)|mid[- ]sized compan",
 "lending.corporate.sme": r"small (and medium|business)|\bsmes?\b|small[- ]business (loans?|lending)|small companies .{0,20}(loans?|credit)",
 "lending.leveraged_sponsor.leveraged_loan": r"leveraged loans?|leveraged lending|term loan b|\btlb\b|covenant[- ]lite",
 "lending.leveraged_sponsor.acquisition": r"(acquisition|buyout|lbo|takeover) (financing|loans?|debt)|leveraged buyout|financing (for|of) the (acquisition|takeover|purchase)",
 "lending.leveraged_sponsor.bridge": r"bridge (loans?|financing|facilit\w+)",
 "lending.specialised.project_finance": r"project financ\w+|non[- ]recourse|limited[- ]recourse",
 "lending.specialised.object_finance": r"aircraft financ\w+|ship financ\w+|shipping (loans?|lend\w+)|aircraft (loans?|leas\w+)|rail ?car (financ|leas)",
 "lending.specialised.commodity_finance": r"reserve[- ]based lending|trade financ\w+ .{0,30}(commodit|oil|metals?)|commodity financ\w+|pre[- ]export financ",
 "lending.cre.general": r"commercial (real estate|property) (loans?|lend\w+|debt)|commercial mortgages?|\bcre (loans?|lending|exposure)",
 "lending.cre.ipre": r"(office|hotel|mall|shopping cent\w+|apartment) (loans?|mortgages?)|refinanc\w+ .{0,30}(tower|building|hotel)",
 "lending.cre.hvcre": r"construction (loans?|lending|financ\w+)|development (loans?|financ\w+)|land loans?",
 "lending.fi.banks": r"interbank (loans?|lending)|loans? to (other )?banks|credit lines? to (other )?banks",
 "lending.fi.nbfis": r"(loans?|credit (lines?|facilit\w+)) to (insurers?|finance companies|asset managers?|non[- ]bank)",
 "lending.fi.funds": r"subscription (lines?|facilit\w+|financ\w+)|capital[- ]call (lines?|facilit\w+)|\bnav (loans?|facilit\w+|lending)|fund financ\w+",
 "lending.sovereign_public": r"(loans?|credit) to (the )?(government|state|municipal\w*|sovereign)|sovereign loans?|syndicated loan .{0,40}(government|ministry|republic)",
 "lending.trade.trade_loans": r"trade financ\w+|export (credit|financ\w+|loans?)|import (financ\w+|loans?)",
 "lending.trade.lc_guarantees": r"letters? of credit|bank guarantees?|standby (lc|letters?)",
 "lending.trade.receivables_supply_chain": r"supply[- ]chain financ\w+|factoring|receivables? financ\w+|invoice financ\w+",
 "lending.consumer.cards": r"credit[- ]card (loans?|balances?|lend\w+|charge[- ]offs?|delinquenc\w+|receivables)|card (loans?|lending)",
 "lending.consumer.auto": r"auto loans?|car loans?|auto lend\w+|auto financ\w+|subprime auto",
 "lending.consumer.personal": r"personal loans?|consumer loans?|installment loans?|payday (loans?|lend\w+)",
 "lending.mortgage.owner_occupied": r"(home|residential) (loans?|mortgages?)|mortgage (lending|lenders?|approvals?|originations?|rates?)|first[- ]time buyers?",
 "lending.mortgage.income_producing": r"buy[- ]to[- ]let|rental[- ]property mortgages?|investor mortgages?",
 "lending.mortgage.home_equity": r"home[- ]equity (loans?|lines?)|\bhelocs?\b",
}
BANK = r"\b(bank|banks|lender|lenders|dealer|dealers|goldman|jpmorgan|morgan stanley|citigroup|bank of america|barclays|hsbc|deutsche bank|bnp|ubs|credit suisse|wells fargo|societe generale|mizuho|mufg|nomura)\b"
df = pl.read_parquet("corpus/canonical.parquet").select(pl.concat_str([pl.col("headline"), pl.col("article")], separator=" ").str.to_lowercase().alias("t"))
n = df.height
t0 = time.time()
df = df.with_columns(pl.col("t").str.contains(BANK).alias("_bank"))
rows = []
for k, pat in T.items():
    m = df["t"].str.contains(pat)
    rows.append((k, int(m.sum()), int((m & df["_bank"]).sum())))
print(f"{n} articles, bank/dealer mention in {df['_bank'].mean():.1%} ({time.time()-t0:.0f}s)\n")
print(f"{'node':45s} {'term hits':>10s} {'%':>6s} {'+bank':>8s} {'%':>6s}")
for k, a, b in sorted(rows, key=lambda r: r[2]):
    print(f"{k:45s} {a:10,d} {a/n:6.2%} {b:8,d} {b/n:6.2%}")
json.dump({k: [a, b] for k, a, b in rows}, open("ontology/term_prevalence_v1.0.json", "w"))
