# -*- coding: utf-8 -*-
"""
Loc va lam sach du lieu cho de tai:
"Phan tich nhu cau su dung giuong benh va du bao luong benh nhan nhap vien khan cap"

Cach dung:
  1. Dat 3 file CSV vao thu muc setup/ cua repo, doi ten thanh:
       patients.csv        (ho so benh nhan)
       bed_occupancy.csv   (ty le lap day giuong benh)
       ae_2010_2020.csv    (so lieu cap cuu/nhap vien 2010-2020)
  2. Chay tu thu muc goc cua repo:  python scr/loc_du_lieu.py
     (hoac mo bang VS Code va bam "Run Cell" o tung o "# %%")
  3. Ket qua luu o setup/: patients_clean.csv, bed_occupancy_clean.csv,
     ae_clean.csv, ae_monthly.csv, ae_benhvien.csv
     daily_patients.csv
     (daily_patients.csv = chuoi THEO NGAY: nhap vien + so giuong dang dung,
      tu file patients -> file chinh de huan luyen mo hinh du bao;
      ae_benhvien.csv = chuoi theo thang cua 1 benh vien, dung doi chieu;
      doi benh vien o muc 4, bien BENH_VIEN)
"""

# %% 0. Thu vien va duong dan
from pathlib import Path

import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # chay tung o trong VS Code / Jupyter
    ROOT = Path.cwd()

DATA_DIR = ROOT / "setup"
print("Thu muc du lieu:", DATA_DIR)


# %% 1. File ho so benh nhan: patients.csv
bn = pd.read_csv(DATA_DIR / "patients.csv")
print("1.1 Kich thuoc ban dau:", bn.shape)

# 1.2 Chuan hoa kieu du lieu
bn["arrival_date"] = pd.to_datetime(bn["arrival_date"], format="%m/%d/%Y", errors="coerce")
bn["departure_date"] = pd.to_datetime(bn["departure_date"], format="%m/%d/%Y", errors="coerce")
bn["service"] = bn["service"].str.strip()
bn["name"] = bn["name"].str.strip()

# 1.3 Loc du lieu loi
n0 = len(bn)
bn = bn.drop_duplicates(subset="patient_id")
bn = bn.dropna(subset=["patient_id", "arrival_date", "departure_date"])
bn = bn[bn["departure_date"] >= bn["arrival_date"]]
bn = bn[bn["age"].between(0, 120)]
bn = bn[bn["satisfaction"].between(0, 100)]
print(f"1.3 Da loai {n0 - len(bn)} dong loi, con lai {len(bn)} dong")

# 1.4 Them cot so ngay nam vien
bn["length_of_stay"] = (bn["departure_date"] - bn["arrival_date"]).dt.days

bn.to_csv(DATA_DIR / "patients_clean.csv", index=False, encoding="utf-8-sig")
print(bn.describe(include="all").T[["count", "min", "max"]])


# %% 2. File ty le lap day giuong benh: bed_occupancy.csv
gb = pd.read_csv(DATA_DIR / "bed_occupancy.csv")
print("2.1 Kich thuoc ban dau:", gb.shape)

# 2.2 Chuan hoa ngay, sap xep, bo trung
gb["Date"] = pd.to_datetime(gb["Date"], format="%m/%d/%Y", errors="coerce")
gb = gb.dropna(subset=["Date", "Occupancy"]).drop_duplicates(subset="Date")
gb = gb.sort_values("Date").reset_index(drop=True)

# 2.3 Tinh lai cac cot lich tu cot Date de dam bao dung
gb["Day_of_Week"] = gb["Date"].dt.dayofweek
gb["Month"] = gb["Date"].dt.month
gb["Year"] = gb["Date"].dt.year

# 2.4 Kiem tra chuoi thoi gian lien tuc va gia tri hop le
so_ngay_thieu = (gb["Date"].diff().dt.days.dropna() != 1).sum()
print("2.4 So cho bi nhay ngay:", so_ngay_thieu)
gb = gb[gb["Occupancy"] >= 0]

# 2.5 Tinh lai cot tre (lag) tu Occupancy; giu gia tri goc o cac dong dau
gb["Prev_Occupancy"] = gb["Occupancy"].shift(1).fillna(gb["Prev_Occupancy"]).astype(int)
gb["Prev_2_Occupancy"] = gb["Occupancy"].shift(2).fillna(gb["Prev_2_Occupancy"]).astype(int)

gb.to_csv(DATA_DIR / "bed_occupancy_clean.csv", index=False, encoding="utf-8-sig")
print("2.5 Con lai", len(gb), "dong, tu", gb["Date"].min().date(), "den", gb["Date"].max().date())


# %% 3. File cap cuu / nhap vien 2010-2020: ae_2010_2020.csv
ae = pd.read_csv(DATA_DIR / "ae_2010_2020.csv")
print("3.1 Kich thuoc ban dau:", ae.shape)

PCT = "Percentage in 4 hours or less (all)"
ae = ae.drop(columns=["Unnamed: 0"], errors="ignore")  # cot chi so thua
ae["date"] = pd.to_datetime(ae["date"], format="%m/%d/%Y")


# %% 3.2 Sua cac cot month, year, phan tram bi nhan len nhieu lan
# Cot year bi nhan voi he so 1-5 (vi du 2012 -> 4024). Cot month va ty le
# phan tram cung bi nhan cung he so do. Chia lai de khoi phuc.
he_so = ae["year"] / ae["date"].dt.year
print("3.2 Cac he so bi nhan:", sorted(he_so.round().unique()))
ae[PCT] = ae[PCT] / he_so
ae["year"] = ae["date"].dt.year
ae["month"] = ae["date"].dt.month


# %% 3.3 Thong nhat ten co so y te
# Cung mot noi nhung viet khac nhau (Children'S / Children's, WiC...).
# Gom theo chu thuong, lay cach viet xuat hien nhieu nhat lam ten chuan.
ae = ae.dropna(subset=["Name"]).copy()  # 4 dong khong co ten (deu bang 0)
ae["Name"] = ae["Name"].str.strip()
khoa = ae["Name"].str.lower().str.replace(r"\s+", " ", regex=True)
cach_viet = ae.groupby([khoa.rename("khoa"), "Name"]).size().reset_index(name="so_dong")
cach_viet = cach_viet.sort_values(["khoa", "so_dong"], ascending=[True, False])
ten_chuan = cach_viet.drop_duplicates("khoa").set_index("khoa")["Name"]
n_truoc = ae["Name"].nunique()
ae["Name"] = khoa.map(ten_chuan)
print(f"3.3 So ten co so: {n_truoc} -> {ae['Name'].nunique()}")


# %% 3.4 Loai dong trung (cung co so, cung thang)
# Sau khi thong nhat ten co the xuat hien dong trung. Giu dong co
# Total attendances lon nhat.
n0 = len(ae)
ae = ae.sort_values("Total attendances", ascending=False)
ae = ae.drop_duplicates(subset=["Name", "date"], keep="first")
ae = ae.sort_values(["Name", "date"]).reset_index(drop=True)
print("3.4 Da loai", n0 - len(ae), "dong trung")


# %% 3.5 Xu ly gia tri am va kieu so nguyen
cot_so_luong = [c for c in ae.columns if c not in ["date", "Name", "month", "year", "lat", "lon", PCT]]
ae[cot_so_luong] = ae[cot_so_luong].clip(lower=0).round().astype("int64")  # -1 -> 0
ae[PCT] = ae[PCT].clip(0, 1)


# %% 3.6 Loc dong khong bao cao (Total attendances = 0)
# Cac dong nay la co so chua mo / khong gui so lieu, khong phai "0 ca".
n0 = len(ae)
ae = ae[ae["Total attendances"] > 0].reset_index(drop=True)
print("3.6 Da loai", n0 - len(ae), "dong khong bao cao, con", len(ae), "dong")


# %% 3.7 Sua toa do lat/lon sai va dien thieu
# Mot so co so bi gan toa do o My (lon < -8), ngoai nuoc Anh -> dat thanh NaN.
ngoai_nuoc_anh = ~ae["lat"].between(49.8, 60.9) | ~ae["lon"].between(-8.7, 1.8)
ae.loc[ngoai_nuoc_anh, ["lat", "lon"]] = np.nan
# Dien toa do thieu bang toa do khac cua cung co so
ae["lat"] = ae.groupby("Name")["lat"].transform("first")
ae["lon"] = ae.groupby("Name")["lon"].transform("first")
print("3.7 So dong con thieu toa do:", ae["lat"].isna().sum())

ae.to_csv(DATA_DIR / "ae_clean.csv", index=False, encoding="utf-8-sig")


# %% 3.8 Gop thanh chuoi thoi gian theo thang (dung cho ARIMA / Prophet)
NHAP_VIEN = [
    "Emergency Admissions via Type 1 A&E in 4 hours",
    "Emergency Admissions via Type 2 A&E in 4 hours",
    "Emergency Admissions via Type 3 and 4 A&E in 4 hours",
    "Other Emergency admissions (i.e not via A&E)",
]
ae["emergency_admissions"] = ae[NHAP_VIEN].sum(axis=1)

thang = (
    ae.groupby("date")
    .agg(
        attendances=("Total attendances", "sum"),
        emergency_admissions=("emergency_admissions", "sum"),
        n_org=("Name", "nunique"),  # so co so co bao cao trong thang
    )
    .reset_index()
)
thang.to_csv(DATA_DIR / "ae_monthly.csv", index=False, encoding="utf-8-sig")
print(thang.head())
print("3.8 Chuoi theo thang:", len(thang), "thang, tu",
      thang["date"].min().date(), "den", thang["date"].max().date())


# %% 4. Loc theo de tai: chon 1 benh vien cu the
# Doi 3 dong duoi day neu muon chon benh vien hoac khoang thoi gian khac.
BENH_VIEN = "Nottingham University Hospitals NHS Trust"
TU_NGAY = "2016-01-01"
DEN_NGAY = "2019-12-01"  # dung o 12/2019 de tranh anh huong dich COVID-19 nam 2020

# 4.1 Danh sach benh vien phu hop: co khoa Cap cuu lon (Type 1) va bao cao du
# tat ca cac thang trong khoang thoi gian tren
so_thang = len(pd.date_range(TU_NGAY, DEN_NGAY, freq="MS"))
khung = ae[
    (ae["date"] >= TU_NGAY)
    & (ae["date"] <= DEN_NGAY)
    & (ae["Type 1 Departments - Major A&E"] > 0)
]
ung_vien = khung.groupby("Name").agg(
    so_thang=("date", "nunique"),
    nhap_vien_trung_binh=("emergency_admissions", "mean"),
)
ung_vien = ung_vien[ung_vien["so_thang"] == so_thang]
ung_vien = ung_vien.sort_values("nhap_vien_trung_binh", ascending=False)
print(f"4.1 Co {len(ung_vien)} benh vien du {so_thang} thang. 10 benh vien lon nhat:")
print(ung_vien.head(10).round(0).to_string())


# %% 4.2 Lay du lieu cua benh vien da chon
if BENH_VIEN not in ung_vien.index:
    raise SystemExit(f"'{BENH_VIEN}' khong du du lieu. Chon ten khac trong danh sach o 4.1.")

bv = ae[(ae["Name"] == BENH_VIEN) & (ae["date"] >= TU_NGAY) & (ae["date"] <= DEN_NGAY)]
bv = bv.sort_values("date").reset_index(drop=True)
print("4.2 Benh vien:", BENH_VIEN, "-", len(bv), "thang")


# %% 4.3 Phat hien va sua thang bat thuong
# Mot so thang (vi du 04/2019, 06/2015) co so lieu gap doi binh thuong o rat
# nhieu benh vien. Thang nao lon hon 1.6 lan trung vi cac thang xung quanh
# duoc danh dau va thay bang noi suy tuyen tinh.
trung_vi = bv["Total attendances"].rolling(7, center=True, min_periods=3).median()
bv["is_outlier"] = (bv["Total attendances"] / trung_vi) > 1.6
print("4.3 Cac thang bat thuong:", bv.loc[bv["is_outlier"], "date"].dt.strftime("%m/%Y").tolist())

cot_dem = cot_so_luong + ["emergency_admissions"]
bv[cot_dem] = bv[cot_dem].astype("float64")
bv.loc[bv["is_outlier"], cot_dem] = np.nan
bv[cot_dem] = bv[cot_dem].interpolate(method="linear", limit_direction="both").round().astype("int64")


# %% 4.4 Them cot thoi gian va luu file
bv["quarter"] = bv["date"].dt.quarter
bv = bv[["date", "Name", "year", "quarter", "month", "Type 1 Departments - Major A&E",
         "Total attendances", "emergency_admissions", "is_outlier"]]
bv.to_csv(DATA_DIR / "ae_benhvien.csv", index=False, encoding="utf-8-sig")
print(bv.describe().T[["count", "min", "max"]])


# %% 5. Chuoi thoi gian THEO NGAY tu ho so benh nhan (khop de tai)
# De tai yeu cau tong hop luot nhap vien theo gio/ngay va du bao nhu cau giuong.
# File patients co ngay den va ngay ve cua tung benh nhan nen tinh duoc:
#   - admissions: so benh nhan nhap vien moi ngay (theo tung khoa/dich vu)
#   - occupied_beds: so giuong dang co nguoi nam moi ngay
NGAY_BAT_DAU = "2025-01-01"
NGAY_KET_THUC = "2025-12-31"
QUOC_GIA_NGAY_LE = "VN"  # ma quoc gia de lay ngay le; doi neu can (can: py -m pip install holidays)

# 5.1 Dem so nhap vien, xuat vien moi ngay
moc = pd.date_range(
    min(pd.Timestamp(NGAY_BAT_DAU), bn["arrival_date"].min()),
    bn["departure_date"].max(),
    freq="D",
)
nhap = bn.groupby("arrival_date").size().reindex(moc, fill_value=0)
xuat = bn.groupby("departure_date").size().reindex(moc, fill_value=0)

# So giuong dang dung cuoi ngay = tong nhap vien - tong xuat vien tinh den ngay do
giuong = nhap.cumsum() - xuat.cumsum()

# 5.2 Nhap vien theo tung khoa/dich vu (emergency, ICU, surgery, general_medicine)
theo_khoa = (
    pd.crosstab(bn["arrival_date"], bn["service"])
    .reindex(moc, fill_value=0)
    .add_prefix("admissions_")
)

ngay = pd.DataFrame({"admissions": nhap, "discharges": xuat, "occupied_beds": giuong})
ngay = ngay.join(theo_khoa)
ngay = ngay.loc[NGAY_BAT_DAU:NGAY_KET_THUC].rename_axis("date").reset_index()

# 5.3 Danh dau giai doan khoi dong: dau nam chua co benh nhan nam tu truoc nen
# so giuong tang gia tao. Bo qua cac ngay nay khi huan luyen mo hinh.
so_ngay_khoi_dong = int(bn["length_of_stay"].max())
ngay["warm_up"] = ngay["date"] < (bn["arrival_date"].min() + pd.Timedelta(days=so_ngay_khoi_dong))

# 5.4 Them cac yeu to co the tac dong den luong benh nhan
ngay["day_of_week"] = ngay["date"].dt.dayofweek
ngay["month"] = ngay["date"].dt.month
ngay["is_weekend"] = (ngay["day_of_week"] >= 5).astype(int)
try:
    import holidays

    ngay_le = holidays.country_holidays(QUOC_GIA_NGAY_LE, years=ngay["date"].dt.year.unique().tolist())
    ngay["is_holiday"] = ngay["date"].dt.date.map(lambda d: int(d in ngay_le))
except Exception as loi:  # chua cai thu vien holidays
    print("5.4 Chua lay duoc ngay le (", loi, ") -> is_holiday = 0. Cai bang: py -m pip install holidays")
    ngay["is_holiday"] = 0

ngay.to_csv(DATA_DIR / "daily_patients.csv", index=False, encoding="utf-8-sig")
print(f"5.4 Chuoi theo ngay: {len(ngay)} ngay, {int(ngay['warm_up'].sum())} ngay khoi dong")
print(ngay[~ngay["warm_up"]][["admissions", "occupied_beds"]].describe().round(1))


# %% 6. Tong ket
print("\nDa luu cac file vao", DATA_DIR)
for ten in ["patients_clean", "bed_occupancy_clean", "ae_clean", "ae_monthly",
            "ae_benhvien", "daily_patients"]:
    print(" -", ten + ".csv")