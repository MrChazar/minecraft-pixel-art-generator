import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';

@Injectable()
export class EmailService {
    constructor(private readonly mailService: MailerService) {}


    async sendVerificationMail(email:string, verifyUrl:string){
        await this.mailService.sendMail({
            to: email,
            subject: 'Verify your email',
            template: './verify-email', // automatically adds .hbs
            context: { name: 'User', verifyUrl, year: new Date().getFullYear() },
            from: '"Generator Frajdy" <noreply@yourdomain.com>',
        })
    }
    async sendPasswordResetMail(email:string, resetUrl:string){
        await this.mailService.sendMail({
            to: email,
            subject: 'Reset your password',
            template: './reset-password',
            context: { name:  'User', resetUrl, year: new Date().getFullYear() },
            from: '"Generator Frajdy" <noreply@yourdomain.com>',
        })
    }
}
