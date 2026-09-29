"""Generate health-insurance parser fixtures."""
from pathlib import Path
import pandas as pd


def create_samples(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    claims = pd.DataFrame({'Claim ID':['0012','0013','0014','0015','0016'],
        'Service Code':['DEMO-SESSION','DEMO-VISIT','DEMO-REMOTE','NA','DEMO-VISIT'],
        'Billed Amount':[220,150,75,None,0], 'Currency':['USD']*5,
        'Units':[2,1,1,1,1], 'Product':['Meadow']*5})
    claims.to_csv(target/'insurance_claims.csv',index=False)
    with pd.ExcelWriter(target/'insurance_mappings.xlsx') as writer:
        pd.DataFrame([
            ['Bingle-Dingle Insurance — attribute mappings'],
            ['Purpose: compare claim intake attributes across intake systems.'],
            ['Scope: claim intake attributes and service references.'],
            ['Missing billed amount differs from a zero charge.'],
            ['Service code definitions are listed in Service Codes.'],
            ['Identifiers must preserve leading zeros.'],
            ['Sources are test inputs, not final payment decisions.'],
            [None], [None], ['Late note: amounts are in USD; no currency conversion is needed.'],
        ]).to_excel(writer,sheet_name='About',index=False,header=False)
        pd.DataFrame([['Version','Change'],['1.0','Initial intake mappings'],
            ['2.0','Added rendering provider identifier'],['2.1','Clarified missing billed amounts'],
        ]).to_excel(writer,sheet_name='Version History',index=False,header=False)
        header=['Source Field','Canonical Field','Status','Notes']
        pd.DataFrame([
            ['Meadow intake mapping'], ['Intake export'], [None], header,
            ['claim_no','claim_id','Approved','Preserve zeros'],
            ['member_no','member_id','Approved','Preserve member identifiers'],
            ['charge','billed_amount','Approved','USD'], header,
            ['svc_date','service_date','Approved','Date of service'], [None],
            ['render_id','rendering_provider_id','Approved','Separate from billing provider'],
            ['auth_ref','authorization_id','Draft','Review required'],
            ['Note: draft mappings need review'],
        ]).to_excel(writer,sheet_name='Meadow Mapping',index=False,header=False)
        pd.DataFrame([header,['id','claim_id','Approved',''],['member','member_id','Approved',''],
            ['amount','billed_amount','Approved','USD'],['amount','billed_amount','Approved','USD'],
            ['date','service_date','Approved',''],['provider','rendering_provider_id','Draft',''],
            ['auth','authorization_id','Draft',''],
        ]).to_excel(writer,sheet_name='River Mapping',index=False,header=False)
        pd.DataFrame([['Source Field','Canonical Field','Notes','Notes'],
            ['claim_id','claim_id','Text identifier','Do not strip zeros'],
        ]).to_excel(writer,sheet_name='Provider Mapping',index=False,header=False)
        claims.to_excel(writer,sheet_name='Claim Samples',index=False)
        pd.DataFrame({'Code':['DEMO-VISIT','DEMO-SESSION','DEMO-REMOTE','NA'],
            'Description':['Office evaluation','Outpatient physical therapy','Virtual office visit','Literal source code, not a missing cell']}
        ).to_excel(writer,sheet_name='Service Codes',index=False)
        pd.DataFrame({'Field':['claim_id','billed_amount'],
            'Definition':['Text claim identifier','Submitted charge in USD; not allowed or paid amount']}
        ).to_excel(writer,sheet_name='Attribute Dictionary',index=False)
    return target/'insurance_mappings.xlsx'


if __name__ == '__main__':
    print(create_samples('data'))
