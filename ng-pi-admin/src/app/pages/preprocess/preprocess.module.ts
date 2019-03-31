import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { routing } from './preprocess.routing';
import { SharedModule } from '../../shared/shared.module';
import { PreprocessComponent } from './preprocess.component';
import { HttpClientModule } from "@angular/common/http";
import { FormsModule } from '@angular/forms'
import { NgxPaginationModule } from 'ngx-pagination';

@NgModule({
    imports: [
        NgxPaginationModule,
        CommonModule,
        SharedModule,
        routing,
        HttpClientModule,
        FormsModule
    ],
    declarations: [
        PreprocessComponent
    ]
})
export class PreprocessModule { }
