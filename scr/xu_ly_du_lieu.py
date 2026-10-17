# -*- coding: utf-8 -*-
"""
LOC, LAM SACH, DICH SANG TIENG VIET VA TAO BO DU LIEU MO HINH (GOP 1 FILE)
De tai: "Phan tich nhu cau su dung giuong benh va du bao luong benh nhan nhap vien khan cap"

Cach dung:
  1. Cai thu vien:  pip install pandas numpy holidays
  2. Dat 3 file CSV vao thu muc setup/ cua repo, doi ten thanh:
       patients.csv        (ho so benh nhan)
       bed_occupancy.csv   (ty le lap day giuong benh)
       ae_2010_2020.csv    (so lieu cap cuu/nhap vien 2010-2020)
  3. Chay tu thu muc goc cua repo:  python scr/xu_ly_du_lieu.py
     (hoac mo bang VS Code va bam "Run Cell" o tung o "# %%")

Cac buoc:
  Muc 1-5  : loc va lam sach du lieu (file *_clean.csv, ae_monthly.csv, ae_benhvien.csv, daily_patients.csv)
  Muc 6-9  : dich sang tieng Viet (file *_vi.csv), them yeu to ngay le/cuoi tuan/mua, thong ke nhanh
  Muc 10   : tao bo du lieu cho mo hinh (setup/du_lieu_mo_hinh.csv) - FILE CHINH de huan luyen
             ARIMA / Prophet / Random Forest / XGBoost
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


# %% 6.0 Ham phu tro cho phan dich
def doc(ten_file, **kw):
    return pd.read_csv(DATA_DIR / ten_file, **kw)


def doi_ten_cot(df, tu_dien):
    """Chi doi ten cac cot co trong du lieu, cot khac giu nguyen."""
    return df.rename(columns={k: v for k, v in tu_dien.items() if k in df.columns})


def luu(df, ten_file):
    df.to_csv(DATA_DIR / ten_file, index=False, encoding="utf-8-sig")
    print(f"  -> {ten_file}: {len(df)} dong, {df.shape[1]} cot")


# %% 6. Tu dien dich
THU = {0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm", 4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"}

KHOA = {
    "emergency": "Cấp cứu",
    "ICU": "Hồi sức tích cực",
    "surgery": "Phẫu thuật",
    "general_medicine": "Nội tổng quát",
}

COT_BENH_NHAN = {
    "patient_id": "Mã bệnh nhân", "name": "Họ tên", "age": "Tuổi",
    "arrival_date": "Ngày nhập viện", "departure_date": "Ngày xuất viện",
    "service": "Khoa", "satisfaction": "Điểm hài lòng", "length_of_stay": "Số ngày nằm viện",
}

COT_GIUONG = {
    "Date": "Ngày", "Occupancy": "Số giường đang dùng", "Day_of_Week": "Thứ trong tuần",
    "Month": "Tháng", "Year": "Năm", "Holiday": "Ngày lễ",
    "Prev_Occupancy": "Số giường dùng hôm trước", "Prev_2_Occupancy": "Số giường dùng 2 ngày trước",
}

COT_CAP_CUU = {
    "date": "Ngày", "Name": "Tên cơ sở y tế", "year": "Năm", "quarter": "Quý", "month": "Tháng",
    "lat": "Vĩ độ", "lon": "Kinh độ",
    "Total attendances": "Tổng lượt khám cấp cứu",
    "emergency_admissions": "Nhập viện khẩn cấp",
    "attendances": "Tổng lượt khám cấp cứu",
    "n_org": "Số cơ sở báo cáo",
    "is_outlier": "Tháng bất thường",
    "Type 1 Departments - Major A&E": "Khoa cấp cứu lớn (Loại 1)",
    "Percentage in 4 hours or less (all)": "Tỷ lệ xử lý trong 4 giờ",
    "Emergency Admissions via Type 1 A&E in 4 hours": "Nhập viện qua cấp cứu Loại 1",
    "Emergency Admissions via Type 2 A&E in 4 hours": "Nhập viện qua cấp cứu Loại 2",
    "Emergency Admissions via Type 3 and 4 A&E in 4 hours": "Nhập viện qua cấp cứu Loại 3 và 4",
    "Other Emergency admissions (i.e not via A&E)": "Nhập viện khẩn cấp ngoài cấp cứu",
}

COT_NGAY = {
    "date": "Ngày", "admissions": "Số ca nhập viện", "discharges": "Số ca xuất viện",
    "occupied_beds": "Số giường đang dùng", "warm_up": "Giai đoạn khởi động",
    "day_of_week": "Thứ trong tuần (0=Hai)", "month": "Tháng",
    "is_weekend": "Cuối tuần", "is_holiday": "Ngày lễ",
}
for k, v in KHOA.items():
    COT_NGAY[f"admissions_{k}"] = f"Nhập viện - {v}"


# %% 7. Dich tung file da lam sach
print("7. Dich cac file sang tieng Viet")

# 2.1 Ho so benh nhan (them nhom tuoi de phan tich theo loai benh nhan)
bn = doc("patients_clean.csv", parse_dates=["arrival_date", "departure_date"])
bn["service"] = bn["service"].map(KHOA).fillna(bn["service"])
bn["nhom_tuoi"] = pd.cut(bn["age"], [-1, 17, 59, 120], labels=["Trẻ em (0-17)", "Người lớn (18-59)", "Cao tuổi (60+)"])
bn = bn.sort_values(["arrival_date", "patient_id"]).reset_index(drop=True)
luu(doi_ten_cot(bn, {**COT_BENH_NHAN, "nhom_tuoi": "Nhóm tuổi"}), "patients_vi.csv")

# 2.2 Ty le lap day giuong
gb = doc("bed_occupancy_clean.csv", parse_dates=["Date"]).sort_values("Date")
gb["Day_of_Week"] = gb["Date"].dt.dayofweek.map(THU)
luu(doi_ten_cot(gb, COT_GIUONG), "bed_occupancy_vi.csv")

# 2.3 Cap cuu / nhap vien (ca nuoc, theo co so)
ae = doc("ae_clean.csv", parse_dates=["date"]).sort_values(["Name", "date"])
luu(doi_ten_cot(ae, COT_CAP_CUU), "ae_vi.csv")

# 2.4 Chuoi theo thang
thang = doc("ae_monthly.csv", parse_dates=["date"]).sort_values("date")
luu(doi_ten_cot(thang, COT_CAP_CUU), "ae_monthly_vi.csv")

# 2.5 Chuoi theo thang cua 1 benh vien
bv = doc("ae_benhvien.csv", parse_dates=["date"]).sort_values("date")
luu(doi_ten_cot(bv, COT_CAP_CUU), "ae_benhvien_vi.csv")


# %% 8. Chuoi theo ngay: them yeu to tac dong (ngay le, cuoi tuan, mua)
print("8. Chuoi theo ngay")
ng = doc("daily_patients.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# 3.1 Ngay lien ke ngay le: nguoi dan thuong nhap vien nhieu sau/ truoc ky nghi dai
ng["truoc_le"] = ng["is_holiday"].shift(-1).fillna(0).astype(int)
ng["sau_le"] = ng["is_holiday"].shift(1).fillna(0).astype(int)

# 3.2 Mua (Viet Nam: mua mua 5-10, mua kho 11-4) va quy
ng["mua"] = np.where(ng["month"].between(5, 10), "Mùa mưa", "Mùa khô")
ng["quy"] = ng["date"].dt.quarter
ng["ten_thu"] = ng["day_of_week"].map(THU)

# 3.3 Sap xep cot theo nhom: thoi gian -> muc tieu -> theo khoa -> yeu to tac dong
cot_khoa = [c for c in ng.columns if c.startswith("admissions_")]
thu_tu = (["date", "ten_thu", "day_of_week", "month", "quy", "mua", "warm_up",
           "admissions", "discharges", "occupied_beds"] + cot_khoa
          + ["is_weekend", "is_holiday", "truoc_le", "sau_le"])
ng = ng[thu_tu]

ten_moi = {**COT_NGAY, "ten_thu": "Tên thứ", "quy": "Quý", "mua": "Mùa",
           "truoc_le": "Trước ngày lễ", "sau_le": "Sau ngày lễ"}
luu(doi_ten_cot(ng, ten_moi), "daily_patients_vi.csv")


# %% 9. Phan tich nhanh cac yeu to tac dong (de dua vao bao cao)
print("9. Yeu to tac dong den so ca nhap vien (bo giai doan khoi dong)")
hop_le = ng[~ng["warm_up"]]
print("\n4.1 Trung binh nhap vien theo thu:")
print(hop_le.groupby("ten_thu")["admissions"].mean().reindex(list(THU.values())).round(2).to_string())
print("\n4.2 Ngay le / ngay thuong:")
print(hop_le.groupby("is_holiday")["admissions"].agg(["mean", "count"]).round(2).rename(index={0: "Ngày thường", 1: "Ngày lễ"}).to_string())
print("\n4.3 Theo thang:")
print(hop_le.groupby("month")[["admissions", "occupied_beds"]].mean().round(2).to_string())


# %% 10. Bo du lieu cho mo hinh du bao
# Muc tieu: du bao "so giuong dang dung" (nhu cau giuong) va "so ca nhap vien".
# Dac trung tre (lag) chi dung gia tri cua cac ngay TRUOC de tranh ro ri thong tin.
print("10. Tao bo du lieu cho mo hinh")

# So giuong toi da cua benh vien: file patients khong co thong tin nay.
# DOI con so nay neu biet so giuong thuc te. Neu de None se uoc luong = 1.1 x gia tri lon nhat.
TONG_GIUONG = None
SO_NGAY_KIEM_TRA = 60  # 60 ngay cuoi dung de kiem tra mo hinh (RMSE, MAE)

md = ng[~ng["warm_up"]].copy().reset_index(drop=True)

if TONG_GIUONG is None:
    TONG_GIUONG = int(np.ceil(md["occupied_beds"].max() * 1.1))
    print(f"  Chua dat TONG_GIUONG -> tam uoc luong {TONG_GIUONG} giuong (nen thay bang so thuc te)")
md["tong_giuong"] = TONG_GIUONG
md["giuong_trong"] = (TONG_GIUONG - md["occupied_beds"]).clip(lower=0)
md["ty_le_lap_day"] = (md["occupied_beds"] / TONG_GIUONG).clip(0, 1).round(4)

for k in [1, 2, 7, 14]:
    md[f"giuong_tre_{k}"] = md["occupied_beds"].shift(k)
    md[f"nhap_vien_tre_{k}"] = md["admissions"].shift(k)
md["nhap_vien_tb7"] = md["admissions"].shift(1).rolling(7).mean()
md["giuong_tb7"] = md["occupied_beds"].shift(1).rolling(7).mean()

md = md.dropna().reset_index(drop=True)
md["tap"] = np.where(md.index >= len(md) - SO_NGAY_KIEM_TRA, "kiểm tra", "huấn luyện")

ten_md = {**ten_moi,
          "tong_giuong": "Tổng số giường", "giuong_trong": "Số giường trống",
          "ty_le_lap_day": "Tỷ lệ lấp đầy giường", "nhap_vien_tb7": "Nhập viện TB 7 ngày trước",
          "giuong_tb7": "Giường dùng TB 7 ngày trước", "tap": "Tập dữ liệu"}
for k in [1, 2, 7, 14]:
    ten_md[f"giuong_tre_{k}"] = f"Giường dùng trước {k} ngày"
    ten_md[f"nhap_vien_tre_{k}"] = f"Nhập viện trước {k} ngày"
luu(doi_ten_cot(md, ten_md), "du_lieu_mo_hinh.csv")

print(f"  Huan luyen: {(md['tap'] == 'huấn luyện').sum()} ngay | Kiem tra: {(md['tap'] == 'kiểm tra').sum()} ngay")
print("\nXong. File chinh de huan luyen mo hinh: setup/du_lieu_mo_hinh.csv")


# %% 11. Tong ket
print("\nDa luu cac file vao", DATA_DIR)
for ten in ["patients_clean", "bed_occupancy_clean", "ae_clean", "ae_monthly", "ae_benhvien", "daily_patients",
            "patients_vi", "bed_occupancy_vi", "ae_vi", "ae_monthly_vi", "ae_benhvien_vi",
            "daily_patients_vi", "du_lieu_mo_hinh"]:
    print(" -", ten + ".csv")